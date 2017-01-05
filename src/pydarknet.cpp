// pydarknet.cpp
// 
// From https://raw.githubusercontent.com/BriSkyHekun/py-darknet-yolo/master/src/pydarknet.cpp

#include <iostream>
#include <boost/python.hpp>
#include <Python.h>
#include <vector>

#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"

#ifdef GPU
#include "cuda.h"
#endif

using namespace std;
namespace bp = boost::python;

typedef struct BBox{
	int left;
	int right;
	int top;
	int bottom;
	float confidence;
	int cls;
} _BBox;

void compute_detections_bbox(image im, int num, float thresh, box *boxes, 
	float **probs, int classes, vector<_BBox> &bb, int draw) {
	int i;

	for (i = 0; i < num; ++i){
		int classs = max_index(probs[i], classes);
		float prob = probs[i][classs];
		if (prob > thresh){
			box b = boxes[i];
			
			int left = (b.x - b.w / 2.)*im.w;
			int right = (b.x + b.w / 2.)*im.w;
			int top = (b.y - b.h / 2.)*im.h;
			int bot = (b.y + b.h / 2.)*im.h;
			
			if (left < 0) left = 0;
			if (right > im.w - 1) right = im.w - 1;
			if (top < 0) top = 0;
			if (bot > im.h - 1) bot = im.h - 1;
			
			_BBox bs;
			bs.left = left; 
			bs.right = right; 
			bs.top = top; 
			bs.bottom = bot; 
			bs.confidence = prob; 
			bs.cls = classs;
			bb.push_back(bs);
			
			if(draw) {
				int width = pow(prob, 1. / 2.) * 10 + 1;
				int offset = classs * 17 % classes;
				float red = get_color(0, offset, classes);
				float green = get_color(1, offset, classes);
				float blue = get_color(2, offset, classes);			
				draw_box_width(im, left, top, right, bot, width, red, green, blue);
				save_image(im, "predictions");
			}
		}
	}
}

class DarknetObjectDetector{

private:
	float thresh;
	float nms;
	bool draw;
	
	network net;
	layer l;
	
	box *boxes;
	float **probs;

public:
	DarknetObjectDetector(bp::str cfg_name, bp::str weight_name, float thresh_, float nms_, int draw_){
		
		std::cout << thresh_ << std::endl;
		
		nms = nms_;
		thresh = thresh_;
		draw = draw_;

		// Load config
		string cfg_c_name = string(((const char*)bp::extract<const char*>(cfg_name)));
		net = parse_network_cfg((char*)cfg_c_name.c_str());
		
		// Load weights
		string weight_c_name = string(((const char*)bp::extract<const char*>(weight_name)));
		load_weights(&net, (char*)weight_c_name.c_str());
		
		// Configure network
		set_batch_network(&net, 1); 
		srand(2222222);
		l = net.layers[net.n-1];
		
		// Configure storage
		boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
		probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
		for (int j = 0; j < l.w*l.h*l.n; j++) {
			probs[j] = (float *)calloc(l.classes, sizeof(float));
		}
	};

	bp::list detect_object(bp::str img_data, int img_width, int img_height, int img_channel){
		bp::list ret_list = bp::list();
		vector<_BBox> bboxes;

		// Load image
		const unsigned char* data = (const unsigned char*)((const char*)bp::extract<const char*>(img_data));
		assert(img_channel == 3);
		image im = make_image(img_width, img_height, img_channel);
		int cnt = img_height * img_channel * img_width;
		for (int i = 0; i < cnt; ++i){
			im.data[i] = (float)data[i] / 255.;
		}
		image sized = resize_image(im, net.w, net.h);
		
		// Predict
		network_predict(net, sized.data);
		
		// Get + filter boxes
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
		
		// Draw
		compute_detections_bbox(im, l.w*l.h*l.n, thresh, boxes, probs, l.classes, bboxes, draw);

		// Clean up        
		free_image(im);
		free_image(sized);
		
		// Return 
		for (int i = 0; i < bboxes.size(); i++) {
			ret_list.append<BBox>(bboxes[i]);
		}
		return ret_list;
	};

	static void set_device(int dev_id) {
#ifdef GPU
		cudaError_t err = cudaSetDevice(dev_id);
		if (err != cudaSuccess){
			cout << "CUDA Error on setting device: " << cudaGetErrorString(err) << '\n';
			PyErr_SetString(PyExc_Exception, "Not able to set device");
		}
#else
		PyErr_SetString(PyExc_Exception, "Not compiled with CUDA");
#endif
	}
	
	~DarknetObjectDetector() {
		free(boxes);
		for (int j = 0; j < l.w*l.h*l.n; j++)
		{
			free(probs[j]);
		}
		free(probs);
	};
};

BOOST_PYTHON_MODULE(libpydarknet)
{
	bp::class_<DarknetObjectDetector>("DarknetObjectDetector", bp::init<bp::str, bp::str, float, float, int>())
		.def("detect_object", &DarknetObjectDetector::detect_object)
		.def("set_device", &DarknetObjectDetector::set_device)
		.staticmethod("set_device");

	bp::class_<BBox>("BBox")
		.def_readonly("left", &BBox::left)
		.def_readonly("right", &BBox::right)
		.def_readonly("top", &BBox::top)
		.def_readonly("bottom", &BBox::bottom)
		.def_readonly("confidence", &BBox::confidence)
		.def_readonly("cls", &BBox::cls);
}
