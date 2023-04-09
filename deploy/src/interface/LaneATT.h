#pragma once
#if defined (_WINDOWS)
  // windows api import/export
  #ifndef LANEATT_IMPORTS
    #define LANEATT_API  __declspec(dllexport)
  #else
    #define LANEATT_API  __declspec(dllimport)
  #endif
#else
  // linux api
  #define LANEATT_API __attribute__((visibility("default")))
#endif

#define IN
#define OUT
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

enum LineType
{
	TYPE_LINE_NOT_SURE = -1,				// 不确定类型
	TYPE_LINE_SPECIAL = 0,					// 特殊类型
	TYPE_LINE_SINGLE_SOLID_WHITE = 1,		// 单白实线
	TYPE_LINE_SINGLE_SOLID_YELLOW = 2,		// 单黄实线
	TYPE_LINE_SINGLE_DASH_WHITE = 3,		// 单白虚线
	TYPE_LINE_SINGLE_DASH_YELLOW = 4,		// 单黄虚线
	TYPE_LINE_DOUBLE_SOLID_WHITE = 5,		// 双白实线
	TYPE_LINE_DOUBLE_SOLID_YELLOW = 6,		// 双黄实线
	TYPE_LINE_DOUBLE_DASH_WHITE = 7,		// 双白虚线
	TYPE_LINE_DOUBLE_DASH_YELLOW = 8,		// 双黄虚线

};

struct Lane_Info
{
	// 点坐标
	std::vector<cv::Point> lane_pts;

	// confidence score
	float conf_score = 0.0f;

	// 线型
	LineType line_type = LineType::TYPE_LINE_NOT_SURE;
};

/******************************************************************************************************************
* 功  能：LaneATT_Init函数, 车道线检测初始化函数
* 参  数：( IN/输入参数)：
						handle			单线程空句柄指针
		 (OUT/输出参数)：
						handle			指向lane detector对象的指针
* 返回值：0(正确);非0(不正确)
* 备  注：
******************************************************************************************************************/
LANEATT_API int LaneATT_Init(IN void **handle, std::string model_path);

/******************************************************************************************************************
* 功  能：LaneATT_Detect函数, 车道线检测
* 参  数：( IN/输入参数)：
						handle					单线程句柄指针
						input_image				单帧图像
						session_detection		车道线检测模型
		 (OUT/输出参数)：
						visual_image		visual information
						output_pts			output points
* 返回值：0(正确);非0(不正确)
* 备  注：
******************************************************************************************************************/
LANEATT_API int LaneATT_Detect(IN void *handle,
							   IN cv::Mat& input_image,
							   OUT cv::Mat& visual_image,
							   OUT std::vector<Lane_Info> & process_result);

/******************************************************************************************************************
* 功  能：LaneATT_Uinit函数, 车道线检测反始化函数
* 参  数：( IN/输入参数)：
						handle			单线程句柄指针
		 (OUT/输出参数)：

* 返回值：0(正确);非0(不正确)
* 备  注：
******************************************************************************************************************/
LANEATT_API int LaneATT_Uinit(IN void *handle);
