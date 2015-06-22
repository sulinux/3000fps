#include "LBFRegressor.h"
#include <string>
#include <direct.h>
using namespace std;
using namespace cv;

void detectAndDraw(string path,
				   Mat& img,
				   CascadeClassifier& nestedCascade, LBFRegressor& regressor,
				   double scale, bool tryflip ,double num, int name);

int FaceDetectionAndAlignment( string path, const char* VideoPath){

	extern string cascadeName;
	string inputName;
	Mat frame, image;
	bool tryflip = false;
	double scale  = 1.3;
	CascadeClassifier cascade;

	LBFRegressor regressor;
	regressor.Load(path,"LBF.model");

	if( !cascade.load( path+cascadeName ) ){
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}
	/*    namedWindow( "result", 1 );*/

	VideoCapture capture(VideoPath);
	bool flags = capture.isOpened();
	double fps = capture.get(  CAP_PROP_FPS  );
	double count = capture.get(  CAP_PROP_FRAME_COUNT  );
	int name = 1;
	while(count)
	{
		count = count-1;
		capture>>frame;
		int cols = frame.cols;
		int rows = frame.rows;
		
		if(!frame.empty())
		{   
			resize(frame,image,Size(cols/2,rows/2));
			detectAndDraw( path, image, cascade,regressor, scale, tryflip , fps , name);
			name++;
		}
		else
		{
			cout<<"Process End"<<endl;
			break;
		}
	}

	return 0;
}


void detectAndDraw(string path, 
				   Mat& img, CascadeClassifier& cascade,
				   LBFRegressor& regressor,
				   double scale, bool tryflip ,double count, int name){
					   int i = 0;
					   double t = 0;
					   vector<Rect> faces,faces2;
					   const static Scalar colors[] =  { CV_RGB(0,0,255),
						   CV_RGB(0,128,255),
						   CV_RGB(0,255,255),
						   CV_RGB(0,255,0),
						   CV_RGB(255,128,0),
						   CV_RGB(255,255,0),
						   CV_RGB(255,0,0),
						   CV_RGB(255,0,255)} ;
					   Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

					   cvtColor( img, gray, CV_BGR2GRAY );
					   resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
					   equalizeHist( smallImg, smallImg );

					   // --Detection
					   t = (double)cvGetTickCount();
					   cascade.detectMultiScale( smallImg, faces,
						   1.1, 2, 0
						   |CV_HAAR_SCALE_IMAGE
						   ,
						   Size(30, 30) );
					   if( tryflip ){
						   flip(smallImg, smallImg, 1);
						   cascade.detectMultiScale( smallImg, faces2,
							   1.1, 2, 0
							   |CV_HAAR_SCALE_IMAGE
							   ,
							   Size(30, 30) );
						   for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
						   {
							   faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
						   }
					   }
					   t = (double)cvGetTickCount() - t;
					   printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

					   // --Alignment
					   t =(double)cvGetTickCount();
					   for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ ){
						   Point center;
						   Scalar color = colors[i%8];
						   BoundingBox boundingbox;

						   boundingbox.start_x = r->x*scale;
						   boundingbox.start_y = r->y*scale;
						   boundingbox.width   = (r->width-1)*scale;
						   boundingbox.height  = (r->height-1)*scale;
						   boundingbox.centroid_x = boundingbox.start_x + boundingbox.width/2.0;
						   boundingbox.centroid_y = boundingbox.start_y + boundingbox.height/2.0;

						   string foldername = path + "/videoResult";
						   mkdir(foldername.c_str());
						   imwrite(path+"/videoResult/images_"+to_string(name)+"_Video.jpg",img);
						   t =(double)cvGetTickCount();
						   Mat_<double> current_shape = regressor.Predict(gray,boundingbox,1);
						   t = (double)cvGetTickCount() - t;
						   printf( "alignment time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

						   rectangle(img, cvPoint(boundingbox.start_x,boundingbox.start_y),
							   cvPoint(boundingbox.start_x+boundingbox.width,boundingbox.start_y+boundingbox.height),Scalar(0,255,0), 1, 8, 0);

						   for(int i = 0;i < global_params.landmark_num;i++){
							   circle(img,Point2d(current_shape(i,0),current_shape(i,1)),1,Scalar(0,0,255),3,8,0);
						   }

						   ofstream ptsfile;
						   ptsfile.open(path + "/videoResult/images_"+to_string(name)+"_Video.pts");
						   ptsfile<<"version: 1"<<endl;
						   ptsfile<<"n_points: "<<global_params.landmark_num<<endl;
						   ptsfile<<'{'<<endl;
						   for(int i = 0;i < global_params.landmark_num;i++)
						   {
							   ptsfile<<current_shape(i,0)<<" "<<current_shape(i,1)<<endl;
						   }
						   ptsfile<<'}'<<endl;
						   ptsfile.close();
						     
					   }
					   cv::imshow( "result", img );
					   waitKey(1000/count);
}
