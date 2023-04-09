#include "LaneATT.h"
#include <laneATT_model.h>

using laneATT::laneATT_detection::laneATT_model;

LANEATT_API int LaneATT_Init(IN void **handle, std::string model_path)
{

	laneATT_model * detector = new laneATT_model(model_path);

	if (detector == NULL)
	{
		std::cout << "输入的Handle指针为空" << std::endl;
		return -1;
	}

	*handle = (void *)detector;

	return 0;
}

LANEATT_API int LaneATT_Detect(IN void *handle,
							   IN cv::Mat& input_image,
							   OUT cv::Mat& visual_image,
							   OUT std::vector<Lane_Info> & process_result)
{

	laneATT_model * detector = (laneATT_model *)handle;
	detector->detect(input_image, visual_image, process_result);
	return 0;

}


LANEATT_API int LaneATT_Uinit(IN void * handle)
{

	laneATT_model * detector = (laneATT_model *)handle;
	detector->~laneATT_model();
	return 0;

}
