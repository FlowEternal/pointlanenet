// ——————————————————————————————————————————————————————————————————————————————
// File Name	:laneATT_model.cpp
// Abstract 	:laneATT
// Version  	:1.0
// Author		:zhan dong xu
// Date			:2021/03/15
// ——————————————————————————————————————————————————————————————————————————————

#include "laneATT_model.h"

std::vector<cv::Scalar>  line_color_list = { 
	cv::Scalar(0,0,255),			// 不确定类型
	cv::Scalar(255,0,255),			// 特殊类型
	cv::Scalar(255,255,255),		// 单白实线
	cv::Scalar(0,255,255),			// 单黄实线
	cv::Scalar(255,255,255),		// 单白虚线
	cv::Scalar(0,255,255),			// 单黄虚线
	cv::Scalar(255,255,255),		// 双白实线
	cv::Scalar(0,255,255),			// 双黄实线
	cv::Scalar(255,255,255),		// 双白虚线
	cv::Scalar(0,255,255),			// 双黄虚线
};

std::vector<std::string>  line_type_list = {
	"uncertain",
	"special",
	"ssolid-w",
	"ssolid-y",
	"sdash-w",
	"sdash-y",
	"dsolid-w",
	"dsolid-y",
	"ddash-w",
	"ddash-y"
};


// ———————————————————————————————————
// ———————————显示点相关参数———————————
// ———————————————————————————————————
#define VISUALIZATION_BOX		false
#define OUTPUT_PT				false
#define VISUAL_POINT_RADIUS		3
#define VISUAL_POINT_WIDTH		-1
#define VISUAL_POINT_COLOR		cv::Scalar(0,255,0)

// ———————————————————————————————————
// ———————————显示线相关的参数———————————
// ———————————————————————————————————
// line显示参数
#define VISUAL_LINE_WIDTH	2					
// box显示参数
#define WIDTH_BOX_BASE		float(20)		
#define HEIGHT_BOX_BASE		float(15)	
#define BOX_COLOR			cv::Scalar(255,255,0)
#define TEXT_SCALE			2					// "Lane"有四个字符
#define BOX_IOU_THRESHOLD	0.000001			// 不让box重合
// 文本显示参数
#define FONT_SCALE_TXT		0.5				
#define THICKNESS_TXT		1					
#define FONT_TYPE			cv::FONT_HERSHEY_COMPLEX
#define TEXT_COLOR			cv::Scalar(0,0,0)
// 虚线相关
#define RATIO_INTERPOLATE	0.5

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length);

// line iou
inline bool devIoU(Lane a, Lane b, const float threshold);

void nms_boxes(std::vector<Lane> & lanes_input);

// box iou
float cal_iou(cv::Rect rect1, cv::Rect rect2);

bool is_overlap_with_any(std::vector<cv::Rect> box_list, cv::Rect target_rect);

inline int get_max_score_index(int & max_index, float & max_score, std::array<float, LANE_TYPE_NUM> input_array);

#if defined (_WINDOWS)
wchar_t * char2wchar(const char* cchar)
{
	wchar_t *m_wchar;
	int len = MultiByteToWideChar(CP_ACP, 0, cchar, strlen(cchar), NULL, 0);
	m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(CP_ACP, 0, cchar, strlen(cchar), m_wchar, len);
	m_wchar[len] = '\0';
	return m_wchar;
}
#endif

namespace laneATT
{
	namespace laneATT_detection
	{
		laneATT_model::laneATT_model(std::string model_path)
		{

			env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Default");
			Ort::SessionOptions session_option;
			session_option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

#if not defined (_WINDOWS)
			if (USING_TENSORRT_ARM)
			{
				Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_option, 0));
			}
#endif
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));


			// 测量模型加载时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(1) Start Model Loading" << std::endl;

#if defined (_WINDOWS)
			const ORTCHAR_T* model_path_convert = char2wchar(model_path.c_str());
			session_ = Ort::Session(env_, model_path_convert, session_option);

#else
			session_ = Ort::Session(env_, model_path.c_str(), session_option);
#endif 


			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Model Loading Time Cost: " << time_used << "ms!" << std::endl;
			std::cout << std::endl;
			// 测量模型加载时间 end

		}

		void laneATT_model::detect(const cv::Mat& input_image,
			cv::Mat& visual_image,std::vector<Lane_Info> & process_result)
		{

			// 这里首先做一个可视化备份并获取原始图像尺寸
			org_img_height = input_image.rows;
			org_img_width = input_image.cols;
			std::cout << std::endl;
			std::cout << "(1) Start Input Tensor Preprocess" << std::endl;
			// —————————————————————————————————————
			// ——————————— prepare input ———————————
			// —————————————————————————————————————
			cv::Mat input_image_copy;
			input_image.copyTo(input_image_copy);

			// 测量preprocess时间 start
			tic = std::chrono::steady_clock::now();
			preprocess(input_image, input_image_copy, visual_image);
			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Input Tensor Preprocess Time Cost: " << time_used << "ms!" << std::endl;
			// 测量preprocess时间 end



			// 测量填充tensor时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(2) Start Input Tensor Filling" << std::endl;
			float* input_image_ptr = input_image_.data();
#if defined(_WINDOWS)
			fill(input_image_.begin(), input_image_.end(), 0.f);
#else
#endif	
			const int row = INPUT_HEIGHT;
			const int col = INPUT_WIDTH;
			const int channel = INPUT_CHANNEL;

			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < col; j++)
				{
					input_image_ptr[0 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 0]);
					input_image_ptr[1 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 1]);
					input_image_ptr[2 * row*col + i * col + j] = (input_image_copy.ptr<float>(i)[j * 3 + 2]);
				}
			}

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Input Tensor Filling Time Cost: " << time_used << "ms!" << std::endl;
			// 测量填充tensor时间 end




			// 测量创建ORT Tensor时间 start
			tic = std::chrono::steady_clock::now();
			std::cout << std::endl;
			std::cout << "(3) Start Tensor Convert" << std::endl;

			// ——————————————————————————————
			// ——————————— tensor ———————————
			// ——————————————————————————————
			const char* input_names[] = { INPUT_DETECTION_NAME };
			const char* output_names[] = { OUTPUT_DETECTION_REG,OUTPUT_DETECTION_CLS,OUTPUT_CLASSIFICATION };

			
			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
			input_image_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				input_image_.data(),
				input_image_.size(),
				input_image_shape_.data(),
				input_image_shape_.size());

			std::vector<Ort::Value> inputs_tensor;
			std::vector<Ort::Value> outputs_tensor;
			inputs_tensor.push_back(std::move(input_image_tensor_));

			// output
			pred_cls_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				pred_cls_.data(),
				pred_cls_.size(),
				pred_cls_shape_.data(),
				pred_cls_shape_.size());

			// output
			pred_reg_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				pred_reg_.data(),
				pred_reg_.size(),
				pred_reg_shape_.data(),
				pred_reg_shape_.size());


			// output
			pred_lane_type_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
				pred_lane_type_.data(),
				pred_lane_type_.size(),
				pred_lane_type_shape_.data(),
				pred_lane_type_shape_.size());

			outputs_tensor.push_back(std::move(pred_reg_tensor_));
			outputs_tensor.push_back(std::move(pred_cls_tensor_));
			outputs_tensor.push_back(std::move(pred_lane_type_tensor_));

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Tensor Convert Time Cost: " << time_used << "ms!" << std::endl;
			// 测量创建ORT Tensor时间 end


			// Single Forward Inference start
			tic = std::chrono::steady_clock::now();

			std::cout << std::endl;

			std::cout << "(4) Start Single Forward Inference" << std::endl;

			session_.Run(Ort::RunOptions{ nullptr },
				input_names,
				inputs_tensor.data(), inputs_tensor.size(),
				output_names,
				outputs_tensor.data(), outputs_tensor.size());

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--Single Forward Inference Time Cost: " << time_used << "ms!" << std::endl;
			// Single Forward Inference end


			// PostProcess start
			tic = std::chrono::steady_clock::now();

			std::cout << std::endl;

			std::cout << "(5) Start PostProcessing" << std::endl;

			// ————————————————————————————
			// ———————— 处理输出张量 ————————
			// ————————————————————————————
			// 创建结果矩阵并进行后处理
			float* output_reg_ptr = outputs_tensor[0].GetTensorMutableData<float>();
			float* output_cls_ptr = outputs_tensor[1].GetTensorMutableData<float>();
			float* output_lane_type_ptr = outputs_tensor[2].GetTensorMutableData<float>();
			postprocess(output_cls_ptr, output_reg_ptr, output_lane_type_ptr, process_result, visual_image);

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac - tic).count();
			std::cout << "-- Summary--PostProcessing Time Cost: " << time_used << "ms!" << std::endl;
			// Postprocess end

		}

		void laneATT_model::preprocess(const cv::Mat &input_image, cv::Mat& output_image, cv::Mat & visual_img)
		{

			// start
			tic_inner = std::chrono::steady_clock::now();

			if (input_image.size() != _m_input_node_size_host)
			{
				cv::resize(input_image, output_image, _m_input_node_size_host, 0, 0, cv::INTER_LINEAR);
			}


			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Image Resize Time Cost: " << time_used << "ms!" << std::endl;
			std::cout << "-- Original Image Size: Height = " << input_image.size().height << ", Width = " << input_image.size().width << std::endl;
			std::cout << "-- Resized Image Size: Height = " << output_image.size().height << ", Width = " << output_image.size().width << std::endl;
			// end


			// start
			tic_inner = std::chrono::steady_clock::now();
			visual_img = output_image.clone();
			if (output_image.type() != CV_32FC3)
			{
				// 首先转化为RGB
				cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
				// 然后转化为float32
				output_image.convertTo(output_image, CV_32FC3); 
				// 进行normalization
				cv::divide(output_image, cv::Scalar(255.0f, 255.0f, 255.0f), output_image);
				cv::subtract(output_image, cv::Scalar(0.485, 0.456, 0.406), output_image);
				cv::divide(output_image, cv::Scalar(0.229, 0.224, 0.225), output_image);
			}

			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Converting Resized Image To Float32 Time Cost: " << time_used << "ms!" << std::endl;
			// end


		}


		void laneATT_model::postprocess(float* output_cls_ptr, 
										float* output_reg_ptr, 
										float* output_lane_type_ptr,
										std::vector< Lane_Info > & process_result,
										cv::Mat & visual_image)
		{

			// start
			tic_inner = std::chrono::steady_clock::now();
			std::vector<Lane>		choose_lane_info;

			for (int index_hegiht = 0; index_hegiht < FEATURE_HEIGHT; index_hegiht++)
				for (int index_width = 0; index_width < FEATURE_WIDTH; index_width++)
				{

					int anchor_index = index_hegiht * FEATURE_WIDTH + index_width;

					// anchor cls
					float * tmp_cls_ptr = output_cls_ptr + anchor_index * CLS_NUM;
					// anchor reg
					float * tmp_reg_ptr = output_reg_ptr + anchor_index * TOTAL_REGRESSION_NUM;
					// anchor lane type
					float * tmp_lane_type_ptr = output_lane_type_ptr + anchor_index * LANE_TYPE_NUM;
					// anchor y pos
					int anchor_y_pos = int((FEATURE_HEIGHT - 1 - index_hegiht) * POINTS_PER_ANCHOR);
					// anchor center x
					float anchor_center_x = (1.0 * index_width + 0.5) * STRIDE_WIDTH;
					// anchor center y
					float anchor_center_y = (1.0 * index_hegiht + 0.5) * STRIDE_HEIGHT;

					// score filter
					float * softmax_array_conf = new float[CLS_NUM];
					activation_function_softmax<float>(tmp_cls_ptr, softmax_array_conf, CLS_NUM);

					// 如果confidence小于阈值 直接pass
					if (softmax_array_conf[CLS_NUM - 1] < conf_threshold)
					{
						delete[] softmax_array_conf;
						continue;
					}

					Lane lane_obj;

					std::vector<cv::Point> lane_pts;

					// up process
					int end_pos = anchor_y_pos;
					int relative_up_end_pos = tmp_reg_ptr[POINTS_PER_LINE + 1];
					for (int up_index = 0; up_index< POINTS_PER_LINE; up_index++)
					{
						// out of range then break
						if ((up_index > relative_up_end_pos) || ((anchor_y_pos + up_index) > POINTS_PER_LINE))
						{
							break;
						}

						float relative_x = tmp_reg_ptr[POINTS_PER_LINE + 2 + up_index] * ANCHOR_INTERVAL; // scale invariable
						float abs_x = relative_x + anchor_center_x; // out

						// out of range break
						if ((abs_x < 0) || (abs_x > NET_DETECTION_INPUT_WIDTH))
						{
							break;
						}

						float abs_y = NET_DETECTION_INPUT_HEIGHT - 1 - (anchor_y_pos + up_index) * ANCHOR_INTERVAL; // out

						// insert pt
						lane_pts.push_back(cv::Point2f(abs_x, abs_y));

						// refresh
						end_pos += 1;

					}

					// down process
					int start_pos = anchor_y_pos;
					int relative_down_end_pos = tmp_reg_ptr[POINTS_PER_LINE];
					for (int down_index = 0; down_index < anchor_y_pos; down_index++)
					{
						// out of range then break
						if ((down_index > relative_down_end_pos) || ((anchor_y_pos -1 - down_index)< 0))
						{
							break;
						}


						float relative_x = tmp_reg_ptr[down_index] * ANCHOR_INTERVAL; // scale invariable
						float abs_x = relative_x + anchor_center_x; // out

						// out of range break
						if ((abs_x < 0) || (abs_x >= NET_DETECTION_INPUT_WIDTH + MARGINE_DOWN_BRANCH))
						{
							break;
						}

						float abs_y = NET_DETECTION_INPUT_HEIGHT - 1 - (anchor_y_pos - 1 - down_index) * ANCHOR_INTERVAL; // out

						// insert pt
						lane_pts.insert(lane_pts.begin(), cv::Point2f(abs_x, abs_y));

						// refresh
						start_pos -=1;

					}

					// wheater total len > 2
					if (lane_pts.size() > MIN_LANE_LENGTH )
					{
						// save 
						lane_obj.index = anchor_index;
						lane_obj.anchor_center_x = anchor_center_x;
						lane_obj.anchor_center_y = anchor_center_y;
						lane_obj.start_pos = start_pos;
						lane_obj.end_pos = end_pos;
						lane_obj.score = softmax_array_conf[CLS_NUM - 1];
						lane_obj.lane_pts = lane_pts;

						// process lane type
						std::string lane_type_name = "";
						float * softmax_array_lane_type = new float[LANE_TYPE_NUM];
						activation_function_softmax<float>(tmp_lane_type_ptr, softmax_array_lane_type, LANE_TYPE_NUM);

						int max_index = 0;
						float max_value = 0.0;
						for (int i = 0; i < LANE_TYPE_NUM; i++)
						{
							if (softmax_array_lane_type[i]> max_value)
							{
								max_value = softmax_array_lane_type[i];
								max_index = i;
							}
						}


						lane_obj.lane_type = get_lane_type(max_index, max_value, lane_type_name);
						lane_obj.lane_type_name = lane_type_name;

						choose_lane_info.push_back(lane_obj);

						delete[] softmax_array_lane_type;

					}


					delete[] softmax_array_conf;

				}


			std::cout << "-- proposal anchor line number before nms: " << choose_lane_info.size() << std::endl;

			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Anchor Thresholding And Preparing before NMS Time Cost: " << time_used << "ms!" << std::endl;
			// end



			// start
			tic_inner = std::chrono::steady_clock::now();
			nms_boxes(choose_lane_info);
			std::cout << "-- proposal anchor line number after nms: " << choose_lane_info.size() << std::endl;

			if (choose_lane_info.size() == 0)
			{
				return;
			}

			int further_rm_num = choose_lane_info.size() - nms_top_k;
			while (further_rm_num-- > 0)
			{
				choose_lane_info.pop_back();
			}

			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Line NMS Time Cost: " << time_used << "ms!" << std::endl;
			// end



			// start
			tic_inner = std::chrono::steady_clock::now();
			for (int index = 0; index < choose_lane_info.size(); index++)
			{
				Lane one_lane_info = choose_lane_info[index];

				Lane_Info output_one_lane_info;

				std::vector<cv::Point> output_one_lane_pts;
				for (int lane_idx = 0; lane_idx < one_lane_info.lane_pts.size() ; lane_idx++)
				{
					// scale back
					int scaled_x_coord = float(one_lane_info.lane_pts[lane_idx].x) / NET_DETECTION_INPUT_WIDTH * org_img_width;
					int scaled_y_coord = float(one_lane_info.lane_pts[lane_idx].y) / NET_DETECTION_INPUT_HEIGHT * org_img_height;
					output_one_lane_pts.push_back(cv::Point(scaled_x_coord, scaled_y_coord));
					continue;

				}

				output_one_lane_info.lane_pts = output_one_lane_pts;
				output_one_lane_info.conf_score = one_lane_info.score;
				output_one_lane_info.line_type = one_lane_info.lane_type;

				process_result.push_back(output_one_lane_info);

			}


			tac_inner = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Decoding Line Information Time Cost: " << time_used << "ms!" << std::endl;
			// end


			// start
			tic_inner = std::chrono::steady_clock::now();

			// —————————————————————————————————————
			// ——————————— 可视化显示车道线 ———————————
			// —————————————————————————————————————
			std::vector<cv::Rect>	box_list;
			for (int lane_idx = 0; lane_idx < process_result.size(); lane_idx++)
			{

				std::vector<cv::Point> & tmp_pts = process_result[lane_idx].lane_pts;

				if (tmp_pts.size() < 2)
				{
					continue;
				}


				// —————————————————————————————————————
				// ——————————— 车道线点连成线 —————————————
				// —————————————————————————————————————
				draw_line(process_result[lane_idx], visual_image);

				std::string info_type = choose_lane_info[lane_idx].lane_type_name; // 这里用choose_lane_info的type
				//std::string info_type = "Lane"; // 这里用choose_lane_info的type

				int width_box = int(WIDTH_BOX_BASE / TEXT_SCALE * (info_type.length()));
				int height_box = int(HEIGHT_BOX_BASE);

				// —————————————————————————————————————————
				// ——————————— 显示车道线的置信度 —————————————
				// —————————————————————————————————————————
				float cof_score_round = process_result[lane_idx].conf_score;
				std::string info_conf = std::to_string(cof_score_round).substr(0, 4);


				// 这里开始进行box的画图
				// 保证每个box不相交
				float text_x = 0;
				float text_y = 0;
				int counter = -1;
				cv::Rect text_box;
				cv::Point pt1;
				cv::Point pt2;

				do
				{

					counter++; // 从0 开始
					if (counter >= process_result[lane_idx].lane_pts.size())
					{
						break;
					}

					text_x = process_result[lane_idx].lane_pts[counter].x / org_img_width * NET_DETECTION_INPUT_WIDTH;
					text_y = process_result[lane_idx].lane_pts[counter].y / org_img_height * NET_DETECTION_INPUT_HEIGHT;
					pt1 = cv::Point(text_x, text_y);
					pt2 = cv::Point(text_x + width_box, text_y - 2 * height_box);
					text_box = cv::Rect(pt1, pt2);

				} while ((text_x + width_box >= NET_DETECTION_INPUT_WIDTH) || (is_overlap_with_any(box_list, text_box)));

				box_list.push_back(text_box);
				cv::rectangle(visual_image, text_box, line_color_list[ int(choose_lane_info[lane_idx].lane_type) + 1], -1); // TODO
				cv::Point text_center_conf = cv::Point(text_x, text_y);
				cv::Point text_center_type = cv::Point(text_x, text_y - height_box);

				// line type
				cv::putText(visual_image, info_conf, text_center_conf,
					FONT_TYPE, FONT_SCALE_TXT,
					TEXT_COLOR, THICKNESS_TXT, 8, 0);

				// confidence
				cv::putText(visual_image, info_type, text_center_type,
					FONT_TYPE, FONT_SCALE_TXT,
					TEXT_COLOR, THICKNESS_TXT, 8, 0);


			}

			tac = std::chrono::steady_clock::now();
			time_used = std::chrono::duration_cast<std::chrono::milliseconds>(tac_inner - tic_inner).count();
			std::cout << "-- Visualization Time Cost: " << time_used << "ms!" << std::endl;
			// end


		}


		LineType laneATT_model::get_lane_type(int max_index, float max_value, std::string & lane_type_name)
		{

			if (max_value > TYPE_THRESHOLD)
			{
				lane_type_name = line_type_list[max_index + 1];
				return (LineType)(max_index);

			}

			else
			{
				lane_type_name = line_type_list[0];
				return LineType::TYPE_LINE_NOT_SURE;
			}


		}


		void laneATT_model::draw_line(Lane_Info & one_lane,cv::Mat & visual_img)
		{
			std::vector<cv::Point> tmp_pts = one_lane.lane_pts;

			for (int idx = 0; idx < tmp_pts.size() - 1; idx++)
			{

				float coord_x_ = tmp_pts[idx].x / org_img_width * NET_DETECTION_INPUT_WIDTH;
				float coord_y_ = tmp_pts[idx].y / org_img_height * NET_DETECTION_INPUT_HEIGHT;
				cv::Point2f pt1 = cv::Point2f(coord_x_, coord_y_);

				float coord_x__ = tmp_pts[idx + 1].x / org_img_width * NET_DETECTION_INPUT_WIDTH;
				float coord_y__ = tmp_pts[idx + 1].y / org_img_height * NET_DETECTION_INPUT_HEIGHT;
				cv::Point2f pt2 = cv::Point2f(coord_x__, coord_y__);

				int x_new = int(pt2.x * RATIO_INTERPOLATE + (1 - RATIO_INTERPOLATE)*pt1.x);
				int y_new = int(pt2.y * RATIO_INTERPOLATE + (1 - RATIO_INTERPOLATE)*pt1.y);
				cv::Point2f pt2_new = cv::Point2f(x_new, y_new);

				switch (one_lane.line_type)
				{

					// 不确定
				case LineType::TYPE_LINE_NOT_SURE:
					cv::line(visual_img, pt1, pt2, line_color_list[0], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);

					// 特殊
				case LineType::TYPE_LINE_SPECIAL:
					cv::line(visual_img, pt1, pt2, line_color_list[1], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);

					// 单白实线
				case LineType::TYPE_LINE_SINGLE_SOLID_WHITE:
					cv::line(visual_img, pt1, pt2, line_color_list[2], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);
					break;

					// 单黄实线
				case LineType::TYPE_LINE_SINGLE_SOLID_YELLOW:
					cv::line(visual_img, pt1, pt2, line_color_list[3], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);
					break;

					// 单白虚线
				case LineType::TYPE_LINE_SINGLE_DASH_WHITE:
					cv::line(visual_img, pt1, pt2_new, line_color_list[4], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);
					break;

					// 单黄虚线
				case LineType::TYPE_LINE_SINGLE_DASH_YELLOW:
					cv::line(visual_img, pt1, pt2_new, line_color_list[5], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);
					break;

				///////////////////////////////////////////////////
					// 双白实线
				case LineType::TYPE_LINE_DOUBLE_SOLID_WHITE:
					cv::line(visual_img, pt1, pt2, line_color_list[6], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);
					break;

					// 双黄实线
				case LineType::TYPE_LINE_DOUBLE_SOLID_YELLOW:
					cv::line(visual_img, pt1, pt2, line_color_list[7], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);
					break;

					// 双白虚线
				case LineType::TYPE_LINE_DOUBLE_DASH_WHITE:
					cv::line(visual_img, pt1, pt2_new, line_color_list[8], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);
					break;

					// 双黄虚线
				case LineType::TYPE_LINE_DOUBLE_DASH_YELLOW:
					cv::line(visual_img, pt1, pt2_new, line_color_list[9], VISUAL_LINE_WIDTH, cv::LineTypes::FILLED);
					break;

				}

			}

		}
	}
}


#pragma region =============================== utility function ===============================

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
	const _Tp alpha = *std::max_element(src, src + length);
	_Tp denominator{ 0 };

	for (int i = 0; i < length; ++i) {
		dst[i] = std::exp(src[i] - alpha);
		denominator += dst[i];
	}

	for (int i = 0; i < length; ++i) {
		dst[i] /= denominator;
	}

	return 0;
}

// IoU部分
inline bool devIoU(Lane a, Lane b, const float threshold) 
{

	int max_start_pos = MAX(a.start_pos, b.start_pos);
	int min_end_pos = MIN(a.end_pos, b.end_pos);

	// quit if no intersection
	if ((min_end_pos <= max_start_pos) || (max_start_pos < 0) || (min_end_pos < 1))
	{
		return false;
	}

	// cal mean dist
	auto pts_a = a.lane_pts;
	auto pts_b = b.lane_pts;
	float dis_mean = 0.0;
	for (int i = max_start_pos; i < min_end_pos ; i++)
	{
		dis_mean += abs(pts_a[i - a.start_pos].x - pts_b[i - b.start_pos].x);

	}

	dis_mean /= (min_end_pos - max_start_pos);

	// cal max distance
	float dis_start = abs(pts_a[max_start_pos - a.start_pos].x - pts_b[max_start_pos - b.start_pos].x);
	float dis_end = abs(pts_a[min_end_pos - 1 - a.start_pos].x - pts_b[min_end_pos - 1 - b.start_pos].x);
	float dis_max = MAX(dis_start, dis_end);
	float dis_another = MAX(dis_mean, dis_max);

	if (USE_MEAN_DISTANCE)
	{
		// based on mean distance
		return (dis_mean > threshold) ? false : true;
	}
	else
	{
		// based on max distance
		return (dis_another > threshold) ? false : true;

	}

}

bool cmpScore(Lane lsh, Lane rsh) 
{
	if (lsh.score > rsh.score)
		return true;
	else
		return false;
}

void nms_boxes(std::vector<Lane> & lanes_input)

{
	Lane lane;
	std::vector<Lane> lanes_process;
	int i, j;
	for (i = 0; i < lanes_input.size(); i++)
	{
		lanes_process.push_back(lanes_input[i]);
	}

	sort(lanes_process.begin(), lanes_process.end(), cmpScore);
	lanes_input.clear();

	int updated_size = lanes_process.size();
	for (i = 0; i < updated_size; i++)
	{

		lanes_input.push_back(lanes_process[i]);
		for (j = i + 1; j < updated_size; j++)
		{
			bool is_suppressed = false;
			is_suppressed = devIoU(lanes_process[i], lanes_process[j], NMS_THRESHOLD);

			if (is_suppressed)
			{
				// 删除掉重复的
				lanes_process.erase(lanes_process.begin() + j);
				updated_size = lanes_process.size();
				j--;
			}

		}
	}


}

// box iou
float cal_iou(cv::Rect rect1, cv::Rect rect2)
{
	//计算两个矩形的交集
	cv::Rect rect_intersect = rect1 & rect2;
	float area_intersect = rect_intersect.area();

	//计算两个举行的并集
	cv::Rect rect_union = rect1 | rect2;
	float area_union = rect_union.area();

	//计算IOU
	double IOU = area_intersect * 1.0 / area_union;

	return IOU;
}

bool is_overlap_with_any(std::vector<cv::Rect> box_list, cv::Rect target_rect)
{

	for (int index = 0; index < box_list.size(); index++)
	{
		float iou = cal_iou(box_list[index], target_rect);
		if (iou > BOX_IOU_THRESHOLD)
		{
			return true;
		}
	}

	return false;

}


inline int get_max_score_index(int & max_index, float & max_score, std::array<float, LANE_TYPE_NUM> input_array)
{
	max_score = 0;
	max_index = 0;
	for (int m = 0; m < LANE_TYPE_NUM; m++)
	{
		if (input_array[m] > max_score)
		{
			max_score = input_array[m];
			max_index = m;
		}
	}

	return 0;
}


#pragma endregion




