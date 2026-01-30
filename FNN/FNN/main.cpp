#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <numeric>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <ctime>
#include <complex>

#include <opencv2/opencv.hpp>

#include "Preprocessing.hpp"
#include "Extraction.hpp"
#include "Network.hpp"
#include "Calculator.hpp"
#include "PolySolver.hpp"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// we start with helpers 
static bool isImageFile(const fs::path& p){
    string ext=p.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext==".png"||ext ==".jpg"||ext==".jpeg"||ext==".bmp");
}
static vector<float> matToFlat28x28(Mat s){
    if (s.empty()) return {};
    if (s.type() != CV_8U) s.convertTo(s, CV_8U);

    vector<Point> pts;
    findNonZero(s, pts);
    if (pts.empty()) return vector<float>(28*28,0.0f);

    Rect box = boundingRect(pts);
    Mat crop = s(box).clone();
    int side = max(crop.cols, crop.rows);
    Mat square = Mat::zeros(side, side, CV_8U);
    crop.copyTo(square(Rect((side-crop.cols)/2, (side-crop.rows)/2,crop.cols, crop.rows)));

    resize(square, square, Size(28,28),0,0,INTER_AREA);
    square.convertTo(square,CV_32F,1.0/255.0f);

    vector<float> flat(784);
    for (int i =0;i<28; i++)
        for (int j=0; j<28;j++)
            flat[i*28+j]=square.at<float>(i,j);
    return flat;
}

static int argmax(const vector<float>& v){
    return (int)(max_element(v.begin(),v.end())-v.begin());
}
static string prettyLabel(const string& folderName){
    if (folderName=="forward_slash") return "/";
    if (folderName=="dot") return "*";
    return folderName;                 
}

struct Dataset{
    vector<vector<float>> X;
    vector<int> y;
};

static bool loadDatasetFromRoot(
    const string& root,
    const vector<string>& classFolders,
    const unordered_map<string,int>& classMap,
    Dataset& out,
    int limitPerClass,          
    bool shuffleFiles,
    mt19937* rng
){
    out.X.clear();
    out.y.clear();
    if (!fs::exists(root)||!fs::is_directory(root)){
        cerr<<"FATAL: dataset root not found: " <<root<<"\n";
        return false;
    }
    for (const auto& folder:classFolders){
        fs::path folderPath=fs::path(root)/folder;
        if (!fs::exists(folderPath)||!fs::is_directory(folderPath)){
            cerr<<"FATAL: missing folder: "<<folderPath.string()<<"\n";
            return false;
        }
        vector<fs::path> files;

        for (const auto& f:fs::directory_iterator(folderPath)){
            if (f.is_regular_file() && isImageFile(f.path()))
                files.push_back(f.path());
        }
        if (files.empty()){
            cerr<<"FATAL: no images in folder: "<<folderPath.string()<<"\n";
            return false;
        }
        if (shuffleFiles && rng) shuffle(files.begin(),files.end(),*rng);
        int take=(limitPerClass<0)?(int)files.size():min(limitPerClass,(int)files.size());
        cout<<"Class '"<< prettyLabel(folder)<<"' -> taking "<< take<<" samples from "<<root<<"\n";

        int classIndex=classMap.at(folder);
        for (int i=0;i<take;i++){
            Mat img=preprocess(files[i].string());
            if (img.empty()) continue;

            vector<float> flat=matToFlat28x28(img);
            if (flat.size()==784){
                out.X.push_back(std::move(flat));
                out.y.push_back(classIndex);
            }
        }
    }
    if (out.X.empty()){
        cerr<<"FATAL: No samples loaded from: "<<root<<"\n";
        return false;
    }
    return true;
}

static float accuracyOnDataset(_NN_& net, const Dataset& d) {
    int correct=0;
    for (size_t i=0; i<d.X.size(); i++) {
        int pred=argmax(net.forward(d.X[i]));
        if (pred==d.y[i])correct++;
    }
    return d.X.empty()?0.0f:(float)correct/(float)d.X.size();
}

int main() {
    auto t_total_start= Clock::now();

    const string trainRoot="data/train";
    const string valRoot="data/validation";
    const string testRoot="data/test";

    //here we can write the name of the img that we want to test
    vector<string> exprCandidates={
        "data/test_images/expr2.png",
        "data/test_images/expr2.jpg",
        "data/test_images/expr2.jpeg"
    };
    // training knobs
    const int LIMIT_PER_CLASS_TRAIN=-1;
    const int LIMIT_PER_CLASS_VAL=-1;
    const int LIMIT_PER_CLASS_TEST=-1;
    const int EPOCHS=80;
    const float LR=0.001f;
    const unsigned SEED=(unsigned)time(nullptr);

    //folders with data which we want to include
    const vector<string> classFolders={
        "0","1","2","3","4","5","6","7","8","9",
        "+","-","dot","forward_slash","x","="
    };

    // we build maps
    unordered_map<string,int> classMap;
    vector<string> indexToLabel;
    indexToLabel.reserve(classFolders.size());
    for (int i=0;i<(int)classFolders.size(); i++){
        classMap[classFolders[i]]=i;
        indexToLabel.push_back(prettyLabel(classFolders[i]));
    }
    const int C=(int)classFolders.size();
    mt19937 rng(SEED);

    // ==================== LOAD TRAIN/VAL/TEST ====================
    Dataset train, val, test;

    if (!loadDatasetFromRoot(trainRoot,classFolders,classMap,train,LIMIT_PER_CLASS_TRAIN,true,&rng))
        return -1;
    if (!loadDatasetFromRoot(valRoot,classFolders,classMap,val,LIMIT_PER_CLASS_VAL,false,nullptr))
        return -1;
    if (!loadDatasetFromRoot(testRoot,classFolders,classMap,test,LIMIT_PER_CLASS_TEST,false,nullptr))
        return -1;

    cout<< "\n================ SPLIT INFO ================\n";
    cout<<"Train root: "<<trainRoot<<"\n";
    cout<<"Val root: "<<valRoot<<"\n";
    cout<<"Test root: "<<testRoot<<"\n";
    cout<<"Train samples: "<<train.X.size()<<"\n";
    cout<<"Val samples: "<<val.X.size()<<"\n";
    cout<<"Test samples: "<<test.X.size()<<"\n";
    cout<<"Classes used: "<<C<<"\n";
    cout<<"Layers: 784 64 32 "<<C<<"\n";
    cout<<"Epochs: "<<EPOCHS<<"\n";
    cout<<"LR: "<<LR<<"\n";
    cout<<"============================================\n\n";
    cout<<"Classes list:\n";
    for (int i=0;i<C;i++) cout<<"  [" <<i<< "] "<<indexToLabel[i]<<"\n";
    cout <<"\n";

    // One-hot targets for TRAIN
    vector<vector<float>> trainTargets(train.X.size(),vector<float>(C,0.0f));
    for (size_t i=0; i<train.y.size();i++) trainTargets[i][train.y[i]]=1.0f;

    // NN
    vector<int> layers = {784,64,32,C};
    _NN_ net(layers);

   // shuffle order for training
    vector<int> order(train.X.size());
    iota(order.begin(), order.end(), 0);

    // accHistory
    vector<float> accHistoryTrain,accHistoryVal,accHistoryTest;
    accHistoryTrain.reserve(EPOCHS);
    accHistoryVal.reserve(EPOCHS);
    accHistoryTest.reserve(EPOCHS);

    // ==================== TRAIN ====================
        auto t_train_start = Clock::now();
        for (int e =0; e<EPOCHS;e++){
            shuffle(order.begin(), order.end(), rng);
            for (size_t t=0;t<order.size();t++){
                int i=order[t];
                net.training(train.X[i],trainTargets[i],LR);
            }
        float accTrain=accuracyOnDataset(net,train);
        float accVal=accuracyOnDataset(net,val);
        float accTest=accuracyOnDataset(net,test);
        accHistoryTrain.push_back(accTrain*100.0f);
        accHistoryVal.push_back(accVal*100.0f);
        accHistoryTest.push_back(accTest*100.0f);

        if (e%10==0||e==EPOCHS-1){
            cout<<"Epoch "<<e<<"/"<<EPOCHS<<" | train="<<accTrain*100.0f<<"% "<<" val="<<accVal*100.0f<< "% "<<" test=" <<accTest*100.0f<<"%\n";
        }
    }
    auto t_train_end=Clock::now();
    cout<<"\nTraining finished.\n";

    // ==================== RANDOM PREDICTIONS ON TEST ====================
    cout <<"\nPredictions(30 random samples):\n";
    for (int t=0;t<30; t++){
        int i=(int)(rng()%test.X.size());
        int pred=argmax(net.forward(test.X[i]));
        cout<<"Sample "<< i<<" true="<<indexToLabel[test.y[i]]<<" pred="<<indexToLabel[pred]<<"\n";
    }

    // ==================== PER-CLASS ACCURACY ON TEST ====================
    vector<int> correctC(C,0),totalC(C,0);
    for (size_t i =0;i<test.X.size();i++){
        int y=test.y[i];
        int pred=argmax(net.forward(test.X[i]));
        totalC[y]++;
        if (pred==y) correctC[y]++;
    }
    cout << "\nPer-class TEST accuracy:\n";
    for (int c=0;c<C;c++){
        double acc=totalC[c] ? (100.0*correctC[c]/totalC[c]):0.0;
        cout<<"Class "<<indexToLabel[c]<<": "<< acc<<"% ("<<correctC[c]<<"/"<<totalC[c]<<")\n";
    }

    // ==================== EXPRESSION TEST ====================
    auto t_expr_start=Clock::now();
    cout<<"\n================ EXPRESSION TEST ================\n";
    string exprPath="";
    for (const auto& p:exprCandidates){
        if (fs::exists(p)){
            exprPath = p; 
            break; 
        }
    }

    string exprPred;
     if (exprPath.empty()){
        cout<<"No expression image found.\n";
        cout<<"Put one of:\n";
        for (auto& p : exprCandidates) cout<<"  - "<<p<<"\n";
    }else {
        Mat exprBin=preprocess(exprPath);
        if (exprBin.empty()){
            cout<<"Cannot read expression image: "<<exprPath<<"\n";
        } else{
            vector<Rect> rects=detectSymbols(exprBin);
            vector<Mat> patches=extractSymbols(exprBin,rects);
            cout<<"Expression file: "<<exprPath<<"\n";
            cout<<"Detected symbols: "<<patches.size()<<"\n";

            //in order to check symbols after running:
            fs::create_directories("debug_patches");

            for (size_t i =0;i<patches.size();i++){
                ostringstream name;
                name<<"debug_patches/patch_"<<setw(3)<<setfill('0')<<i<<".png";
                imwrite(name.str(),patches[i]);

                vector<float> x=matToFlat28x28(patches[i]);
                int pred= argmax(net.forward(x));
                string sym=indexToLabel[pred];
                cout<< "  symbol["<<i<<"] pred='"<<sym<<"'\n";
                exprPred+=sym;
            }

            cout << "Predicted expression: " << exprPred << "\n";
            // for polynomial solver 
            if (exprPred.find('x')!= string::npos || exprPred.find('=') != string::npos){
                vector<complex<double>> roots;
                string polyNorm, perr;

                if (trySolvePolynomial(exprPred, roots, polyNorm, perr)){
                    cout<<"Polynomial detected!\n";
                    cout<<"Roots:\n";
                    for(size_t i=0; i<roots.size();i++)
                        cout<<"  r"<<i+1<<" = "<<roots[i]<<"\n";
                }else {
                    cout<<"Cannot solve polynomial: "<<perr<<"\n";
                }
            } else {
                double value= 0.0;
                string err;
                if (tryEvaluateExpression(exprPred, value,err)){
                    cout<< "Result = "<<value<<"\n";
                } else{
                    cout<<"Cannot evaluate: "<<err<<"\n";
                }
            }
        }
    }

    auto t_expr_end = Clock::now();
    auto t_total_end = Clock::now();

// ---------------- TIMERS -------------------
    cout<<"\n================ TIMERS ================\n";
    cout<< "[Timer] Training took: "<< chrono::duration<double>(t_train_end - t_train_start).count()<<" sec\n";
    cout<<"[Timer] Expression pipeline took: "<< chrono::duration<double>(t_expr_end - t_expr_start).count()<<" sec\n";
    cout<<"[Timer] Total runtime: "<< chrono::duration<double>(t_total_end - t_total_start).count()<<" sec\n";

    return 0;
}
