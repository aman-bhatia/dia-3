#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <vector>

#define HOR 1
#define VER 2
#define coord pair<int,int>		// order is (row,col)

using namespace std;
using namespace cv;


//--------------------------------------------------------Seam Carving--------------------------------------//
struct node{
	int path;	// left is -1, middle is 0, right is 1
	long long int energy;
};

struct by_energy { 
    bool operator()(node const &a, node const &b) { 
        return a.energy < b.energy;
    }
};

Mat seamNormal(Mat img, int cols_to_delete, int mode = VER, bool show = true){
	if (mode == HOR)
		img = img.t();

	if (cols_to_delete > img.cols-2){
		cout << "Too many seams to delete. Give correct parameters.\n";
		exit(0);
	}

	int wait = 60;
	for (int z=0;z<cols_to_delete;z++){

		// Calculate Laplacian
		Mat blured, gray, dst, grad;
		GaussianBlur( img, blured, Size(3,3), 0, 0, BORDER_DEFAULT );
		cvtColor( blured, gray, CV_BGR2GRAY );
		Laplacian( gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT );
		convertScaleAbs( dst, grad );

		// Calculate Sobel
		// Mat gray,grad_x,abs_grad_x,grad_y,abs_grad_y,grad;
		// cvtColor(img,gray,CV_BGR2GRAY);
		// int scale = 1;
		// int delta = 0;
		// int ddepth = CV_16S;
		// Sobel( gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		// convertScaleAbs( grad_x, abs_grad_x );
		// Sobel( gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		// convertScaleAbs( grad_y, abs_grad_y );
		// addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );


		vector< vector<node> > table(0);
		for (int i=0;i<img.rows;i++){
			vector<node> temp(0);
			for (int j=0;j<img.cols;j++){
				node n{0,grad.at<uchar>(i,j)};
				temp.push_back(n);
			}
			table.push_back(temp);
		}

		for (int i=0;i<img.rows;i++){
			for (int j=0;j<img.cols;j++){
				if (i==0){
					continue;
				} else if (j==0){		// extreme left
					long long int mid = table[i-1][j].energy; 
					long long int right = table[i-1][j+1].energy; 
					long long int min_energy = min(mid,right);
					table[i][j].energy += min_energy;
					if (min_energy == mid)
						table[i][j].path = 0;
					else
						table[i][j].path = 1;
				} else if (j==img.cols-1){		// extreme right
					long long int mid = table[i-1][j].energy; 
					long long int left = table[i-1][j-1].energy; 
					long long int min_energy = min(mid,left);
					table[i][j].energy += min_energy;
					if (min_energy == mid)
						table[i][j].path = 0;
					else
						table[i][j].path = -1;
				} else {
					long long int mid = table[i-1][j].energy; 
					long long int left = table[i-1][j-1].energy; 
					long long int right = table[i-1][j+1].energy; 
					long long int min_energy = min(right,min(mid,left));
					table[i][j].energy += min_energy;
					if (min_energy == mid)
						table[i][j].path = 0;
					else if (min_energy == left)
						table[i][j].path = -1;
					else
						table[i][j].path = 1;
				}
			}
		}

		// find the column corresponding to minimum energy
		int min_col = 0;
		long long int min_energy = table[img.rows-1][0].energy;
		for (int i=1;i<img.cols;i++){
			if (table[img.rows-1][i].energy < min_energy){
				min_energy = table[img.rows-1][i].energy;
				min_col = i;
			}
		}

		if (show){
			// paint the corresponding seam RED
			int index = min_col;
			for (int j=img.rows-1; j>=0; j--){
				img.at<Vec3b>(j,index) = {0,0,255};
				index += table[j][index].path;
			}
		}

		if (show){
			if (mode == HOR){
				img = img.t();
			}

			imshow("Seam Carving",img);
			waitKey(wait);
		
			if (mode == HOR){
				img = img.t();
			}
		}

		// remove the corresponding column
		int index = min_col;
		for (int i=img.rows-1;i>=0;i--){
			for (int j=index;j<img.cols-1;j++){
				img.at<Vec3b>(i,j) = img.at<Vec3b>(i,j+1);
			}
			index += table[i][index].path;
		}

		Rect crop_region(0, 0, img.cols-1, img.rows);
		img = img(crop_region);
	}

	if (mode == HOR)
		img =img.t();

	return img;
}



//--------------------------------------------------Pyramids-----------------------------------------------//

Mat viewPyramids(vector<Mat> pyramids){
	int rows = pyramids[0].rows;
	int cols = 0;
	for (int i=0;i<pyramids.size();i++){
		cols += pyramids[i].cols;
	}

	Mat ret = Mat::zeros(rows,cols,pyramids[0].type());

	int col_index = 0;
	for (int i=0;i<pyramids.size();i++){
		pyramids[i].copyTo(ret(Rect(col_index, 0, pyramids[i].cols, pyramids[i].rows)));
		col_index += pyramids[i].cols;
	}
	return ret;
}

// return the downsampled image(at a above level of pyramid)
Mat pyUp(Mat img){
	Mat ret;
	GaussianBlur(img,ret,Size(5,5),0);
	resize(ret,ret,Size(img.cols/2,img.rows/2));
	return ret;
}

vector<Mat> GPyramids(Mat img, int num_levels){
	vector<Mat> ret(1,img);
	for (int i=1;i<num_levels;i++){
		Mat temp_img = pyUp(ret.back());
		ret.push_back(temp_img);
	}
	return ret;
}

vector<Mat> LPyramids(vector<Mat> gaussianPyramids){
	vector<Mat> ret(0);
	for (int i=0;i<gaussianPyramids.size();i++){
		Mat blured, gray, dst, abs_dst;
		GaussianBlur( gaussianPyramids[i], blured, Size(3,3), 0, 0, BORDER_DEFAULT );
		cvtColor( blured, gray, CV_BGR2GRAY );
		Laplacian( gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT );
		convertScaleAbs( dst, abs_dst );
		ret.push_back(abs_dst);
	}
	return ret;
}


//--------------------------------------------------Seam Carving with Pyramids-----------------------------------------------//

pair< Mat , vector< vector< coord > > > seam(Mat img, int cols_to_delete){
	vector< vector< coord > > seams(0);

	if (cols_to_delete > img.cols-2){
		cout << "Too many seams to delete. Give correct parameters.\n";
		exit(0);
	}

	for (int z=0;z<cols_to_delete;z++){
		
		vector< coord > new_seam(0);
		
		// Calculate Laplacian
		Mat blured, gray, dst, grad;
		GaussianBlur( img, blured, Size(3,3), 0, 0, BORDER_DEFAULT );
		cvtColor( blured, gray, CV_BGR2GRAY );
		Laplacian( gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT );
		convertScaleAbs( dst, grad );


		vector< vector<node> > table(0);
		for (int i=0;i<img.rows;i++){
			vector<node> temp(0);
			for (int j=0;j<img.cols;j++){
				node n{0,grad.at<uchar>(i,j)};
				temp.push_back(n);
			}
			table.push_back(temp);
		}

		for (int i=0;i<img.rows;i++){
			for (int j=0;j<img.cols;j++){
				if (i==0){
					continue;
				} else if (j==0){		// extreme left
					long long int mid = table[i-1][j].energy; 
					long long int right = table[i-1][j+1].energy; 
					long long int min_energy = min(mid,right);
					table[i][j].energy += min_energy;
					if (min_energy == mid)
						table[i][j].path = 0;
					else
						table[i][j].path = 1;
				} else if (j==img.cols-1){		// extreme right
					long long int mid = table[i-1][j].energy; 
					long long int left = table[i-1][j-1].energy; 
					long long int min_energy = min(mid,left);
					table[i][j].energy += min_energy;
					if (min_energy == mid)
						table[i][j].path = 0;
					else
						table[i][j].path = -1;
				} else {
					long long int mid = table[i-1][j].energy; 
					long long int left = table[i-1][j-1].energy; 
					long long int right = table[i-1][j+1].energy; 
					long long int min_energy = min(right,min(mid,left));
					table[i][j].energy += min_energy;
					if (min_energy == mid)
						table[i][j].path = 0;
					else if (min_energy == left)
						table[i][j].path = -1;
					else
						table[i][j].path = 1;
				}
			}
		}
		// find the column corresponding to minimum energy
		int min_col = 0;
		long long int min_energy = table[img.rows-1][0].energy;
		for (int i=1;i<img.cols;i++){
			if (table[img.rows-1][i].energy < min_energy){
				min_energy = table[img.rows-1][i].energy;
				min_col = i;
			}
		}

		// remove the corresponding column
		int index = min_col;
		for (int i=img.rows-1;i>=0;i--){
			for (int j=index;j<img.cols-1;j++){
				img.at<Vec3b>(i,j) = img.at<Vec3b>(i,j+1);
			}
			index += table[i][index].path;
		}

		Rect crop_region(0, 0, img.cols-1, img.rows);
		img = img(crop_region);

		seams.push_back(new_seam);
	}
	return make_pair(img , seams);
}



// return image with seams removed to given image
Mat removeSeams(Mat img, vector< vector< coord > > seams){
	Mat ret = img.clone();
	for (int i=0;i<seams.size();i++){
		for (int j=0;j<seams[i].size();j++){
			int row = seams[i][j].first;
			int col = seams[i][j].second;
			for (int k=col;k<img.cols;k++){
				ret.at<Vec3b>(row,k) = ret.at<Vec3b>(row,k+1);
			}
		}
		Rect crop_region(0, 0, ret.cols-1, ret.rows);
		ret = ret(crop_region);
	}
	return ret;
}

// return seams at lower level given seams of upper level
vector< vector< coord > > mapSeams(vector< vector< coord > > seams){
	vector< vector< coord > > seams_ret(0);
	for (int i=0;i<seams.size();i++){
		vector< coord > new_seam(0);
		for (int j=0;j<seams[i].size();j++){
			int a = seams[i][j].first;
			int b = seams[i][j].second;
			new_seam.push_back(make_pair(2*a,2*b));
			new_seam.push_back(make_pair(2*a+1,2*b));
		}
		seams_ret.push_back(new_seam);
		new_seam.clear();
		for (int j=0;j<seams[i].size();j++){
			int a = seams[i][j].first;
			int b = seams[i][j].second;
			new_seam.push_back(make_pair(2*a,2*b+1));
			new_seam.push_back(make_pair(2*a+1,2*b+1));
		}
		seams_ret.push_back(new_seam);
	}
	return seams_ret;
}

Mat seamPyramid(vector<Mat> &gaussians, int cols_to_delete)
{
	int num_levels = gaussians.size();
	vector<int> seam_cols_to_delete(num_levels);
	vector< vector< coord > > seams_deleted_at_previous_level(0);
	for(int l=num_levels-1;l>=0;l--){
		cout << "At level : "<< l << endl;
		int sigma = 0;
		for(int j=l+1;j<num_levels;j++){
			sigma += seam_cols_to_delete[j] * int(pow(2 , j));
		}
		
		seam_cols_to_delete[l] = int((cols_to_delete - sigma) / pow(2 , l));

		// remove seams because of previous level
		vector < vector < coord > > mapped_seams = mapSeams(seams_deleted_at_previous_level);
		gaussians[l] = removeSeams(gaussians[l],mapped_seams);

		pair< Mat , vector< vector< coord > > > result = seam(gaussians[l] , seam_cols_to_delete[l]);
		
		gaussians[l] = result.first;
		vector< vector< coord > > seams_deleted_by_seam_carving = result.second;

		cout << "Num Mapped Seams : " << mapped_seams.size() << endl;
		cout << "Num Seams of Seam Carving : " << seams_deleted_by_seam_carving.size();
		cout << "\n\n";

		seams_deleted_at_previous_level.clear();
		seams_deleted_at_previous_level.insert( seams_deleted_at_previous_level.end(),
												mapped_seams.begin(), mapped_seams.end() );
		
		seams_deleted_at_previous_level.insert( seams_deleted_at_previous_level.end(),
												seams_deleted_by_seam_carving.begin(),
												seams_deleted_by_seam_carving.end() );
	}
	return gaussians[0];

}

int main(int argc , char **argv){
	Mat img = imread(argv[1]);

	int num_levels;
	cout << "Number of pyramid levels : ";
	cin >> num_levels;
	cout << "\n\n";

	vector<Mat> gpy = GPyramids(img, num_levels);
	vector<Mat> lpy = LPyramids(gpy);

	Mat gpyimg = viewPyramids(gpy);
	Mat lpyimg = viewPyramids(lpy);

	imshow("Gaussian Pyramids",gpyimg);
	waitKey();
	destroyAllWindows();
	imshow("Laplacian Pyramids",lpyimg);
	waitKey();
	destroyAllWindows();

	cout << "Rows : " << img.rows << endl;
	cout << "Cols : " << img.cols << endl;

	int cols_to_delete, rows_to_delete;
	cout << "Cols to delete : ";
	cin >> cols_to_delete;
	cout << "Rows to delete : ";
	cin >> rows_to_delete;

	int mode;
	cout << "Choose 1 for normal seam carving, 2 for seam carving with pyramids : ";
	cin >> mode;
	cout << "\n\n";

	float t1=0,t2=0,t3=0,t4=0,t5=0,t6=0;
	if (mode == 1){
		img = seamNormal(img,cols_to_delete,VER,true);
		img = seamNormal(img,rows_to_delete,HOR,true);
		imshow("Seam Carving",img);
		waitKey();
	} else if (mode == 2){
		t1 = clock();
		img = seamPyramid(gpy,cols_to_delete);
		t2 = clock();
		if (rows_to_delete > 0){
			img = img.t();
			vector<Mat> gpy = GPyramids(img, num_levels);
			t3 = clock();
			img = seamPyramid(gpy,rows_to_delete);
			t4 = clock();
			img = img.t();
		}
		imshow("Seam Carving",img);
		imwrite("Seam Carving with pyramids.jpg",img);
		waitKey();
	} else {
		cout << "Choose correct mode next time... Exitting Program.\n\n";
		exit(0);
	}
	destroyAllWindows();
	cout << "Calculating time difference... \n";

	img = imread(argv[1]);
	
	t5 = clock();
	img = seamNormal(img,cols_to_delete,VER,false);
	img = seamNormal(img,rows_to_delete,HOR,false);
	t6 = clock();
	imwrite("Normal Seam Carving.jpg",img);


	if (mode!=2){
		img = imread(argv[1]);
		t1 = clock();
		img = seamPyramid(gpy,cols_to_delete);
		t2 = clock();
		if (rows_to_delete > 0){
			img = img.t();
			vector<Mat> gpy = GPyramids(img, num_levels);
			t3 = clock();
			img = seamPyramid(gpy,rows_to_delete);
			t4 = clock();
			img = img.t();
		}
		imwrite("Seam Carving with pyramids.jpg",img);
	}

	float time1 = (t6 - t5)/CLOCKS_PER_SEC;
	float time2 = (t2-t1 + t4-t3)/CLOCKS_PER_SEC;

	cout << "Time Taken by Normal Seam Carving Method : " << time1 << "\n";
	cout << "Time Taken by Seam Carving with Pyramids : " << time2 << "\n";
	cout << "Reward : " << (time1 - time2) << "\n\n";
	
	return 0;
}