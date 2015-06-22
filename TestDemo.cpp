#include "LBFRegressor.h"
#include <direct.h>
using namespace std;
using namespace cv;
void LoadCofwTestData(vector<Mat_<uchar> >& images,
                      vector<Mat_<double> >& ground_truth_shapes,
                      vector<BoundingBox>& bounding_boxs);
double TestDemo (string path){
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_boxs;
    vector<Mat_<double> >test_ground_truth_shapes;
    int initial_number = 1;

    string testdatapath = path+"/test/Path_Images.txt";
    
	clock_t time = clock();
	int flags = 1;
    LoadOpencvBbxData(flags,path,testdatapath, test_images, test_ground_truth_shapes, test_bounding_boxs);
    
    LBFRegressor regressor;
    regressor.Load(path,"LBF.model");
    vector<Mat_<double> > current_shape = regressor.Predict(test_images,test_bounding_boxs,initial_number);
    
	string name;
	ifstream filename;
	ofstream ptsfile;
	filename.open(testdatapath);
	int k = 0;
	string imStoreName[1000][20];
	string Name;
	while(getline(filename,name))
	{
		int pos1 = name.find_last_of('\\');
		int pos2 = name.find_last_of('\.');
		imStoreName[k][20] = name.substr(pos1,pos2);
		k++;
	}

    double MRSE_sum = 0;
	string foldername = path + "/result";
	mkdir(foldername.c_str());
	cout<<"结果存放的位置：\n"<<foldername<<endl;
    for (int i =0; i<current_shape.size();i++){

//         rectangle(test_images[i], cvPoint(test_bounding_boxs[i].start_x,test_bounding_boxs[i].start_y),
//                   cvPoint(test_bounding_boxs[i].start_x+test_bounding_boxs[i].width,test_bounding_boxs[i].start_y+test_bounding_boxs[i].height),Scalar(0,255,0), 1, 8, 0);
// 
//         for(int j = 0;j < global_params.landmark_num;j++){
//             circle(test_images[i],Point2d(current_shape[i](j,0),current_shape[i](j,1)),1,Scalar(0,0,255),3,8,0);
//         }		
		
		int pos3 = imStoreName[i][20].find_last_of('\.');
		Name = imStoreName[i][20].substr(0,pos3);

		ptsfile.open(path + "/result/"+Name+".pts");
		ptsfile<<"version: 1"<<endl;
		ptsfile<<"n_points: "<<global_params.landmark_num<<endl;
		ptsfile<<'{'<<endl;
		for(int k = 0;k < global_params.landmark_num;k++)
		{
			ptsfile<<current_shape[i](k,0)<<" "<<current_shape[i](k,1)<<endl;
		}
		ptsfile<<'}'<<endl;
		ptsfile.close();

		imwrite(path+"/result/"+imStoreName[i][20],test_images[i]);        
    }
		cout << "test data size: "<<current_shape.size()<<endl;
    
		double time_end = double(clock() - time) / CLOCKS_PER_SEC;
		cout << "the test time  of  "<< current_shape.size()<<" pictures cost "<< time_end<<"s"<<"("<<time_end/current_shape.size()<<" per picture)"<<endl;

		return 0;
}


void LoadCofwTestData(vector<Mat_<uchar> >& images,
                      vector<Mat_<double> >& ground_truth_shapes,
                      vector<BoundingBox>& bounding_boxs){
    int img_num = 507;
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/testImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }
    
    ifstream fin;
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/boundingbox_test.txt");
    for(int i = 0;i < img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0;
        bounding_boxs.push_back(temp);
    }
    fin.close();
    
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/keypoints_test.txt");
    for(int i = 0;i < img_num;i++){
        Mat_<double> temp(global_params.landmark_num,2);
        for(int j = 0;j < global_params.landmark_num;j++){
            fin>>temp(j,0);
        }
        for(int j = 0;j < global_params.landmark_num;j++){
            fin>>temp(j,1);
        }
        ground_truth_shapes.push_back(temp);
    }
    fin.close();
}
