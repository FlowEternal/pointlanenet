// ——————————————————————————————————————————————————————————————————————————————
// File Name	:laneATT_model.h
// Abstract 	:laneATT
// Version  	:1.0
// Author		:zhan dong xu
// Date			:2021/02/26
// ——————————————————————————————————————————————————————————————————————————————
#ifndef ONNX_LaneATT_MODEL_H
#define ONNX_LaneATT_MODEL_H

#include <string>
#include <algorithm>  

#if defined (_WINDOWS)
#include <Windows.h>
#else
#include <sys/time.h> 
#endif

// —————————————————————————
// ———————— 接口头文件 ———————
// —————————————————————————
#include <LaneATT.h>

// ——————————————————————————
// ———————— ONNX头文件 ———————
// ——————————————————————————
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>

#define USING_TENSORRT_ARM true
#if not defined (_WINDOWS)
#if USING_TENSORRT_ARM
#include <tensorrt_provider_factory.h>
#endif
#endif 

// —————————————————————————————————————————
// ———————— Network Detection 宏定义 ————————
// —————————————————————————————————————————
#define INPUT_DETECTION_NAME					"input"
#define OUTPUT_DETECTION_REG					"regression"
#define OUTPUT_DETECTION_CLS					"classification"
#define OUTPUT_CLASSIFICATION					"lane_type"
// 网络输入图像大小
#define NET_DETECTION_INPUT_HEIGHT				288
#define NET_DETECTION_INPUT_WIDTH				512
#define NET_DETECTION_INPUT_CHANNEL				3

// 模型参数设定
#define CLS_NUM									2		// 这里恒定为2
#define LANE_TYPE_NUM							9		// 这里设为车道线类型数

#define ANCHOR_INTERVAL							4
#define POINTS_PER_ANCHOR						4

#define STRIDE									16
#define STRIDE_WIDTH							16
#define STRIDE_HEIGHT							16

#define POINTS_PER_LINE							72
#define FEATURE_WIDTH							int(NET_DETECTION_INPUT_WIDTH/float(STRIDE))
#define FEATURE_HEIGHT							int(NET_DETECTION_INPUT_HEIGHT/float(STRIDE))
#define TOTAL_ANCHOR_NUM						FEATURE_HEIGHT * FEATURE_WIDTH
#define TOTAL_REGRESSION_NUM					(POINTS_PER_LINE + 1) * 2

// 算法参数设定 -- 重要
#define TYPE_THRESHOLD							0.3
#define NMS_THRESHOLD							80
#define USE_MEAN_DISTANCE						false
#define CONF_THRESHOLD_DEFAULT					0.4		// 需要调节
#define LANE_TYPE_THRESHOLD						0.4		// 需要调节 线类型阈值
#define MIN_LANE_LENGTH							2
#define MARGINE_DOWN_BRANCH						100
#define NMS_THRESHOLD_TOP_K						14		// 设定的车道线最大检测数量

// 维度
static constexpr const int INPUT_WIDTH = NET_DETECTION_INPUT_WIDTH;
static constexpr const int INPUT_HEIGHT = NET_DETECTION_INPUT_HEIGHT;
static constexpr const int INPUT_CHANNEL = NET_DETECTION_INPUT_CHANNEL;
static constexpr const int ANCHOR_NUM = TOTAL_ANCHOR_NUM;
static constexpr const int FEATURE_WIDTH_NUM = FEATURE_WIDTH;
static constexpr const int FEATURE_HEIGHT_NUM = FEATURE_HEIGHT;

static constexpr const int Cls_num = CLS_NUM;
static constexpr const int Total_Reg_Num = TOTAL_REGRESSION_NUM;
static constexpr const int Lane_type_num = LANE_TYPE_NUM;

// NMS结构体定义
typedef struct {
	std::vector<cv::Point> lane_pts;
	LineType lane_type;
	std::string lane_type_name;
	float score;
	int start_pos;
	int end_pos;
	float anchor_center_x;
	float anchor_center_y;
	int index;
}Lane;

namespace laneATT 
{
	namespace laneATT_detection 
	{

		class laneATT_model
		{

		public:
			// default constructor
			laneATT_model(std::string model_path);

			void detect(const cv::Mat& input_image,cv::Mat& visual_image,std::vector<Lane_Info> & process_result);

		private:
			// 网络推理引擎
			Ort::Session session_{ nullptr };
			Ort::Env env_{nullptr};

			// 计时器
			std::chrono::steady_clock::time_point tic;
			std::chrono::steady_clock::time_point tac;
			std::chrono::steady_clock::time_point tic_inner;
			std::chrono::steady_clock::time_point tac_inner;
			double time_used = 0;

			// LaneATT的原始输入尺寸
			cv::Size _m_input_node_size_host = cv::Size(INPUT_WIDTH,INPUT_HEIGHT);

			// original image dimension
			float org_img_width = 0.0f;
			float org_img_height = 0.0f;

			// zdx modified 2021/4/2
			// ————————————————————————————————————————————————————————————————————
			// ——————————— Input/Output Tensor Defination for detection ———————————
			// ————————————————————————————————————————————————————————————————————
			// Input image tensor for lane detection
			std::array<float, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNEL> input_image_{};
			std::array<int64_t, 4> input_image_shape_{ 1,INPUT_CHANNEL,INPUT_HEIGHT, INPUT_WIDTH };
			Ort::Value input_image_tensor_{ nullptr };
			
			// output cls
			std::array<float, 1 * ANCHOR_NUM * Cls_num> pred_cls_{};
			std::array<int64_t, 3> pred_cls_shape_{ 1,ANCHOR_NUM, Cls_num };
			Ort::Value pred_cls_tensor_{ nullptr };

			// output reg
			std::array<float, 1 * ANCHOR_NUM * Total_Reg_Num> pred_reg_{};
			std::array<int64_t, 3> pred_reg_shape_{ 1,ANCHOR_NUM, Total_Reg_Num };
			Ort::Value pred_reg_tensor_{ nullptr };

			// ouptut lane type
			std::array<float, 1 * ANCHOR_NUM * Lane_type_num> pred_lane_type_{};
			std::array<int64_t, 3> pred_lane_type_shape_{ 1, ANCHOR_NUM, Lane_type_num };
			Ort::Value pred_lane_type_tensor_{ nullptr };

			// threshold related
			float conf_threshold = CONF_THRESHOLD_DEFAULT;
			float nms_top_k = NMS_THRESHOLD_TOP_K;

			void preprocess(const cv::Mat& input_image, cv::Mat& output_image, cv::Mat &visual_img);

			void postprocess(float* output_cls_ptr, float* output_reg_ptr, float* output_lane_type_ptr,
				std::vector< Lane_Info > & process_result, cv::Mat & visual_image);

			void draw_line(Lane_Info & one_lane, cv::Mat& visual_img);

			LineType get_lane_type(int max_index, float max_value, std::string & lane_type_name);

		};

	}

}

#endif //ONNX_LANEATT_MODEL_H
