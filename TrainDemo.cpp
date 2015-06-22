#include "LBFRegressor.h"
using namespace std;
using namespace cv;
void LoadCofwTrainData(std::string path,
					   vector<Mat_<uchar> >& images,
                       vector<Mat_<double> >& ground_truth_shapes,
                       vector<BoundingBox>& bounding_boxs);
void TrainDemo(string path){
    vector<Mat_<uchar> > images;
    vector<Mat_<double> > ground_truth_shapes;
    vector<BoundingBox> bounding_boxs;

    string traindatapath1 = path+"/train/Path_Images.txt";  
    LBFRegressor regressor;
	int flags = 0;
    LoadOpencvBbxData(flags,path ,traindatapath1, images, ground_truth_shapes, bounding_boxs);
    regressor.Train(images,ground_truth_shapes,bounding_boxs);
	cout<<path+modelPath+"LBF.model"<<endl;
    regressor.Save( path+modelPath, "LBF.model" );
	char conDesign;
	cout<<"ÊÇ·ñÒª²âÊÔ£ºy/n"<<endl;
	cin>>conDesign;
	if ( conDesign == 'y' )
		{
			double MRSE = TestDemo(path);
			cout << "Mean Root Square Error is "<< MRSE*100 <<"%"<<endl;
			return;
		}
	else
       return;
}

void LoadCofwTrainData(std::string path,
					   vector<Mat_<uchar> >& images,
                       vector<Mat_<double> >& ground_truth_shapes,
                       vector<BoundingBox>& bounding_boxs){
    int img_num = 811;
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = path+"/COFW_Dataset/trainingImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }
    
    ifstream fin;
    fin.open("F:/DZP/68_points/COFW_Dataset/boundingbox.txt");
    for(int i = 0;i < img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0; 
        bounding_boxs.push_back(temp);
    }
    fin.close(); 

    fin.open("F:/DZP/68_points/COFW_Dataset/COFW_Dataset/keypoints.txt");
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

