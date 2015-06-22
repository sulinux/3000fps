#include "LBF.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

Params global_params;
string modelPath = "/model/";	
string cascadeName = "/build/haarcascade_frontalface_alt.xml";
int name ;
void InitializeGlobalParam();
void PrintHelp();

int main( int argc, const char** argv )
{   
	   if (argc > 1 && strcmp( argv[1],"TrainModel")==0 ){
        InitializeGlobalParam();
    }
	
	else {
		ReadGlobalParamFromFile( modelPath +"LBF.model" );	
	}

    if (argc==1){
        PrintHelp();
    }
    else if(strcmp(argv[1],"TrainModel")==0){
		
        TrainDemo(argv[2]);
    }
    else if (strcmp(argv[1], "TestModel")==0){
		InitializeGlobalParam();
        double MRSE = TestDemo(argv[2]);
    }
    else if (strcmp(argv[1], "Demo")==0)
	{		
			InitializeGlobalParam();
            return FaceDetectionAndAlignment( argv[2],argv[3] );      
    }
    else {
        PrintHelp();
    }
    return 0;
}

void InitializeGlobalParam(){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 5;
    global_params.landmark_num = 68;
    global_params.initial_num = 5;

    global_params.max_numstage = 7;
    double m_max_radio_radius[10] = {0.4,0.3,0.2,0.15, 0.12, 0.10, 0.08, 0.06, 0.06,0.05};
    double m_max_numfeats[8] = {200,200, 200, 100, 100, 100, 80, 80};
    for (int i=0;i<10;i++){
        global_params.max_radio_radius[i] = m_max_radio_radius[i];
    }
    for (int i=0;i<8;i++){
        global_params.max_numfeats[i] = m_max_numfeats[i];
    }
    global_params.max_numthreshs = 500;
}

void ReadGlobalParamFromFile(string path){
    cout << "Loading GlobalParam..." << endl;
    ifstream fin;
    fin.open(path);
    fin >> global_params.bagging_overlap;
    fin >> global_params.max_numtrees;
    fin >> global_params.max_depth;
    fin >> global_params.max_numthreshs;
    fin >> global_params.landmark_num;
    fin >> global_params.initial_num;
    fin >> global_params.max_numstage;
    
    for (int i = 0; i< global_params.max_numstage; i++){
        fin >> global_params.max_radio_radius[i];
    }
    
    for (int i = 0; i < global_params.max_numstage; i++){
        fin >> global_params.max_numfeats[i];
    }
    cout << "Loading GlobalParam end"<<endl;
    fin.close();
}
void PrintHelp(){
    cout << "Useage:"<<endl;
    cout << "1. Train your own model:    LBF.out.exe  TrainModel dataPath"<<endl;
    cout << "2. Test model on dataset:   LBF.out.exe  TestModel dataPath"<<endl;
    cout << "3. Test model via a camera: LBF.out.exe  Demo VideoPath"<<endl;
    cout << endl;
}