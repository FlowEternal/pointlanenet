"""Utilities and tools for tracking runs with Weights & Biases."""
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))  # add utils/ to path
from utils.datasets import LoadImagesAndLabels
from utils.datasets import img2label_paths
from utils.general import colorstr, xywh2xyxy, check_dataset, check_file

try:
    import wandb
    from wandb import init, finish
except ImportError:
    wandb = None

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def check_wandb_config_file(data_config_file):
    wandb_config = '_wandb.'.join(data_config_file.rsplit('.', 1))  # updated data.yaml path
    if Path(wandb_config).is_file():
        return wandb_config
    return data_config_file


def get_run_info(run_path): # 这里可以看出，wandb的ARTIFACT中，原始yolov5存储的格式是固定好的
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem  # 除去后缀的最后一个组件(查看python官网解释)
    project = run_path.parent.stem # 逻辑路径的父路径
    entity = run_path.parent.parent.stem # 实体提交的
    model_artifact_name = 'run_' + run_id + '_model'
    return entity, project, run_id, model_artifact_name


def check_wandb_resume(opt):
    process_wandb_config_ddp_mode(opt) if opt.global_rank not in [-1, 0] else None  # 根据前缀的情况载入数据 (对于不是单卡训练的-1或者DDP模式的master 0，因为分布式训练的原因需要准备数据)
    if isinstance(opt.resume, str):  # opt.resume默认是Bool参数，这里也就是说命令行必须直接输入
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            if opt.global_rank not in [-1, 0]:  # For resuming DDP runs
                entity, project, run_id, model_artifact_name = get_run_info(opt.resume)
                api = wandb.Api()
                artifact = api.artifact(entity + '/' + project + '/' + model_artifact_name + ':latest')
                modeldir = artifact.download()
                opt.weights = str(Path(modeldir) / "last.pt")  # 下载并获取权重
            return True
    return None


def process_wandb_config_ddp_mode(opt):
    with open(check_file(opt.data)) as f:
        data_dict = yaml.safe_load(f)  # data dict
    train_dir, val_dir = None, None
    if isinstance(data_dict['train'], str) and data_dict['train'].startswith(WANDB_ARTIFACT_PREFIX): # 以'wandb-artifact://'为前缀
        api = wandb.Api()
        train_artifact = api.artifact(remove_prefix(data_dict['train']) + ':' + opt.artifact_alias) # 给wandb的项目取名,opt.artifact_alias对应其别名(查看wandb官网了解详情)
        train_dir = train_artifact.download() # 下载相关数据，并反馈下载的路径
        train_path = Path(train_dir) / 'data/images/'
        data_dict['train'] = str(train_path)

    if isinstance(data_dict['val'], str) and data_dict['val'].startswith(WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        val_artifact = api.artifact(remove_prefix(data_dict['val']) + ':' + opt.artifact_alias)
        val_dir = val_artifact.download()
        val_path = Path(val_dir) / 'data/images/'
        data_dict['val'] = str(val_path)
    if train_dir or val_dir:
        ddp_data_path = str(Path(val_dir) / 'wandb_local_data.yaml')
        with open(ddp_data_path, 'w') as f:
            yaml.safe_dump(data_dict, f)  # 载入相关的配置
        opt.data = ddp_data_path


class WandbLogger():
    """Log training runs, datasets, models, and predictions to Weights & Biases.

    This logger sends information to W&B at wandb.ai. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.

    For more on how this logger is used, see the Weights & Biases documentation:
    https://docs.wandb.com/guides/integrations/yolov5
    """
    def __init__(self, opt, name, run_id, data_dict, job_type='Training'):
        '''
        Args:
            opt: 相关参数选项
            name: logger保存的目录名
            run_id：wandb运行的id
            data_dict：类似于coco128.yaml载入yaml的dict数据
            job_type：工作类型
        
        Returns:
            None
        '''
        # Pre-training routine --
        self.job_type = job_type
        self.wandb, self.wandb_run, self.data_dict = wandb, None if not wandb else wandb.run, data_dict 
        #! wandb.run 是全局变量，默认是None，但是在train.py中wandb_run是进行检查了是否是resume的，因此，如果是resume的，这里是True
        #! 前面train.py的check只是去连接artifact，然后下载last.pt权重文件，而这里是初始化wandb
        # It's more elegant to stick to 1 wandb.init call, but useful config data is overwritten in the WandbLogger's wandb.init call
        if isinstance(opt.resume, str):  # checks resume from artifact 从中断的地方开始训练
            if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                entity, project, run_id, model_artifact_name = get_run_info(opt.resume) #! run_id代表了一次运行的唯一的id
                model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
                assert wandb, 'install wandb to resume wandb runs' #! 这里直接对包进行确认的方式，具有极强的参考意义
                # Resume wandb-artifact:// runs here| workaround for not overwriting wandb.config
                self.wandb_run = wandb.init(id=run_id, project=project, entity=entity, resume='allow') # 定义的相关wandb的初始化参数
                opt.resume = model_artifact_name
        elif self.wandb: # 重新开始一个wandb的project
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",
                                        project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                        entity=opt.entity,
                                        name=name,
                                        job_type=job_type,
                                        id=run_id) if not wandb.run else wandb.run  # 尝试了wandb，确实有点东西，凸(艹皿艹 )！
        if self.wandb_run:
            if self.job_type == 'Training':
                if not opt.resume: # 训练且未中断训练
                    wandb_data_dict = self.check_and_upload_dataset(opt) if opt.upload_dataset else data_dict
                    # Info useful for resuming from artifacts INFO.https://docs.wandb.ai/ref/python/config 代表了训练中的所有参数
                    self.wandb_run.config.opt = vars(opt) #! vars用于返回opt对应类别的参数(初始化或者内部赋值)
                    self.wandb_run.config.data_dict = wandb_data_dict
                self.data_dict = self.setup_training(opt, data_dict)
            if self.job_type == 'Dataset Creation': # 数据集上传与创建到wandb artifact
                self.data_dict = self.check_and_upload_dataset(opt)
        else:
            prefix = colorstr('wandb: ')
            print(f"{prefix}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")

    def check_and_upload_dataset(self, opt):
        assert wandb, 'Install wandb to upload dataset'
        check_dataset(self.data_dict) # 找这个数据，如果没有找到，那么会去通过对应的url，bash或者python脚本去下载
        config_path = self.log_dataset_artifact(check_file(opt.data),
                                                opt.single_cls,
                                                'YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem) 
        #! 这里是把相关数据log上传到wandb(要是数据量很大怎么办？)，config_path就是配置文件的路径
        print("Created dataset config file ", config_path)
        with open(config_path) as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    def setup_training(self, opt, data_dict):
        self.log_dict, self.current_epoch, self.log_imgs = {}, 0, 16  # Logging Constants
        self.bbox_interval = opt.bbox_interval # 记录间隔
        if isinstance(opt.resume, str):
            modeldir, _ = self.download_model_artifact(opt)
            if modeldir:
                self.weights = Path(modeldir) / "last.pt"
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp = str(
                    self.weights), config.save_period, config.total_batch_size, config.bbox_interval, config.epochs, \
                                                                                                       config.opt['hyp']
            data_dict = dict(self.wandb_run.config.data_dict)  # eliminates the need for config file to resume
        if 'val_artifact' not in self.__dict__:  # If --upload_dataset is set, use the existing artifact, don't download #! 如果upload_dataset被设置了，那么验证集和训练集的数据本地都有啊~
            self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(data_dict.get('train'),
                                                                                           opt.artifact_alias) # 数据地址，artifact的handle
            self.val_artifact_path, self.val_artifact = self.download_dataset_artifact(data_dict.get('val'),
                                                                                       opt.artifact_alias)
            self.result_artifact, self.result_table, self.val_table, self.weights = None, None, None, None
            if self.train_artifact_path is not None:
                train_path = Path(self.train_artifact_path) / 'data/images/'
                data_dict['train'] = str(train_path)
            if self.val_artifact_path is not None:
                val_path = Path(self.val_artifact_path) / 'data/images/'
                data_dict['val'] = str(val_path)
                self.val_table = self.val_artifact.get("val")
                self.map_val_table_path()
        if self.val_artifact is not None:
            self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")
            self.result_table = wandb.Table(["epoch", "id", "prediction", "avg_confidence"])
        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
        return data_dict

    def download_dataset_artifact(self, path, alias):
        if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
            artifact_path = Path(remove_prefix(path, WANDB_ARTIFACT_PREFIX) + ":" + alias)
            dataset_artifact = wandb.use_artifact(artifact_path.as_posix())
            assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn\'t exist'"
            datadir = dataset_artifact.download()
            return datadir, dataset_artifact
        return None, None

    def download_model_artifact(self, opt):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(opt.resume, WANDB_ARTIFACT_PREFIX) + ":latest")
            assert model_artifact is not None, 'Error: W&B model artifact doesn\'t exist'
            modeldir = model_artifact.download()
            epochs_trained = model_artifact.metadata.get('epochs_trained')
            total_epochs = model_artifact.metadata.get('total_epochs')
            is_finished = total_epochs is None
            assert not is_finished, 'training is finished, can only resume incomplete runs.'
            return modeldir, model_artifact # 模型路径，artifact字符串路径
        return None, None

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={
            'original_url': str(path),
            'epochs_trained': epoch + 1,
            'save period': opt.save_period,
            'project': opt.project,
            'total_epochs': opt.epochs,
            'fitness_score': fitness_score
        })
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        wandb.log_artifact(model_artifact,
                           aliases=['latest', 'last', 'epoch ' + str(self.current_epoch), 'best' if best_model else ''])
        print("Saving model artifact on epoch ", epoch + 1)

    def log_dataset_artifact(self, data_file, single_cls, project, overwrite_config=False):
        '''
        Args:
            data_file: 类似于coco128.yaml的文件
            single_cls: 对应opt.single_cls(训练多类作为单类?)
            project: wandb对应的项目格式
            overwrite_config: wandb是否覆盖原始配置
        
        Returns:
            path:
        '''
        with open(data_file) as f:
            data = yaml.safe_load(f)  # data dict
        nc, names = (1, ['item']) if single_cls else (int(data['nc']), data['names']) # 类别和名字
        names = {k: v for k, v in enumerate(names)}  # to index dictionary 索引字典
        #! 下面创建wandb对应的table,相关内容的创建非常具有学习意义
        self.train_artifact = self.create_dataset_table(LoadImagesAndLabels( 
            data['train'], rect=True, batch_size=1), names, name='train') if data.get('train') else None
        self.val_artifact = self.create_dataset_table(LoadImagesAndLabels(
            data['val'], rect=True, batch_size=1), names, name='val') if data.get('val') else None
        if data.get('train'):
            data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train')
        if data.get('val'):
            data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')
        path = data_file if overwrite_config else '_wandb.'.join(data_file.rsplit('.', 1))  # updated data.yaml path
        data.pop('download', None)
        with open(path, 'w') as f:
            yaml.safe_dump(data, f)

        if self.job_type == 'Training':  # builds correct artifact pipeline graph
            self.wandb_run.use_artifact(self.val_artifact)
            self.wandb_run.use_artifact(self.train_artifact)
            self.val_artifact.wait() # 这里是等待验证集完成logging
            self.val_table = self.val_artifact.get('val') #! 获取这个表
            self.map_val_table_path()
        else:
            self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.log_artifact(self.val_artifact)
        return path

    def map_val_table_path(self):
        self.val_table_map = {}
        print("Mapping dataset")
        for i, data in enumerate(tqdm(self.val_table.data)):
            self.val_table_map[data[3]] = data[0] # data的格式是["id", "train_image", "Classes", "name"]

    def create_dataset_table(self, dataset, class_to_id, name='dataset'):
        # TODO: Explore multiprocessing to slpit this loop parallely| This is essential for speeding up the the logging
        artifact = wandb.Artifact(name=name, type="dataset")
        img_files = tqdm([dataset.path]) if isinstance(dataset.path, str) and Path(dataset.path).is_dir() else None
        img_files = tqdm(dataset.img_files) if not img_files else img_files
        for img_file in img_files: # 艺术品添加相关的文件和目录
            if Path(img_file).is_dir():
                artifact.add_dir(img_file, name='data/images')
                labels_path = 'labels'.join(dataset.path.rsplit('images', 1))
                artifact.add_dir(labels_path, name='data/labels')
            else:
                artifact.add_file(img_file, name='data/images/' + Path(img_file).name)
                label_file = Path(img2label_paths([img_file])[0])
                artifact.add_file(str(label_file),
                                  name='data/labels/' + label_file.name) if label_file.exists() else None
        table = wandb.Table(columns=["id", "train_image", "Classes", "name"]) #! wandb表格创建
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in class_to_id.items()]) # class_to_id是index和class的键值对
        for si, (img, labels, paths, shapes) in enumerate(tqdm(dataset)):
            box_data, img_classes = [], {}
            for cls, *xywh in labels[:, 1:].tolist():
                cls = int(cls)
                box_data.append({"position": {"middle": [xywh[0], xywh[1]], "width": xywh[2], "height": xywh[3]},
                                 "class_id": cls,
                                 "box_caption": "%s" % (class_to_id[cls])})
                img_classes[cls] = class_to_id[cls]
            boxes = {"ground_truth": {"box_data": box_data, "class_labels": class_to_id}}  # inference-space
            table.add_data(si, wandb.Image(paths, classes=class_set, boxes=boxes), json.dumps(img_classes),
                           Path(paths).name)
        artifact.add(table, name)
        return artifact

    def log_training_progress(self, predn, path, names):
        if self.val_table and self.result_table:
            class_set = wandb.Classes([{'id': id, 'name': name} for id, name in names.items()])
            box_data = []
            total_conf = 0
            for *xyxy, conf, cls in predn.tolist():
                if conf >= 0.25:
                    box_data.append(
                        {"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                         "class_id": int(cls),
                         "box_caption": "%s %.3f" % (names[cls], conf),
                         "scores": {"class_score": conf},
                         "domain": "pixel"})
                    total_conf = total_conf + conf
            boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
            id = self.val_table_map[Path(path).name]
            self.result_table.add_data(self.current_epoch,
                                       id,
                                       wandb.Image(self.val_table.data[id][1], boxes=boxes, classes=class_set),
                                       total_conf / max(1, len(box_data))
                                       )

    def log(self, log_dict):
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        if self.wandb_run:
            wandb.log(self.log_dict)
            self.log_dict = {}
            if self.result_artifact:
                train_results = wandb.JoinedTable(self.val_table, self.result_table, "id")
                self.result_artifact.add(train_results, 'result')
                wandb.log_artifact(self.result_artifact, aliases=['latest', 'last', 'epoch ' + str(self.current_epoch),
                                                                  ('best' if best_result else '')])
                self.result_table = wandb.Table(["epoch", "id", "prediction", "avg_confidence"])
                self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")

    def finish_run(self):
        if self.wandb_run:
            if self.log_dict:
                wandb.log(self.log_dict)
            wandb.run.finish()
