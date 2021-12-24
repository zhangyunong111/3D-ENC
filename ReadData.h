#pragma once

#include <windows.h>
#include <vector>
#include <iostream>
using namespace std;

#include "ogrsf_frmts.h"
//#include <rtreeindex.h>
/*
struct strXY
{
	double dX;
	double dY;
};

class Shape
{
public:
	int _vParts[1000];
	int _iNumParts;
	int _iShapeType;
	double _dBox[4];
	std::vector<strXY> _vPoints;

	virtual void toShape(double, double) = 0;
};

class PointSet :public Shape
{
public:
	~PointSet();
	void toShape(double, double);
};



class LinePolygon :public Shape
{
public:
	int _iNumPnts;
	double _dBox[4];

	virtual void toShape(double, double) = 0;
};


template<typename T>
struct Point {
	int num;//要素编号
	T x;
	T y;
	T z;
	Point() {}
	Point(T _x, T _y, T _z, int _num) :x(_x), y(_y), z(_z), num(_num) {}
};
*/

class RTreeIndex;
class RTreeNode;

template<typename T>
struct Point {
	T x;
	T y;
	T z;
	Point() {}
	Point(T _x, T _y, T _z) :x(_x), y(_y), z(_z) {}
};
/*
template<typename T>
struct MinBoundingBox {
	Point<T> left_up;
	Point<T> right_down;
	//Point<T> center = { (left_up.x + right_down.x) / 2,0.0f,(left_up.z + right_down.z) / 2 };
};*/
//最小MBB
template<typename T>
struct SmallestMinBoundingBox {
	Point<T> p[4];
	double area;
};

template<typename T>
struct Features {
	OGRFeature *ENCFeature;//完整要素
	OGRGeometry *ENCGeometry;//要素几何信息
	OGRFeatureDefn *ENCFeatureDefn;//要素属性
	OGREnvelope ENCenvelope;//几何体范围MBB
	int PointNum;//要素中点的个数
	int FeatureType;//要素类型：0环境要素，1实体要素，2虚拟要素
	int FeatureNum;
	vector<Point<T>> point;
	SmallestMinBoundingBox<T> SMBB;
	/*
	const char *featureID;
	int shapetype;
	int shapenumber;
	MinBoundingBox<T> MBB;
	*/
};

template<typename T>
struct Node {
	Node* R[4];
	Point<T> pt;
	Node* parent;
	//SmallestMinBoundingBox<T> SMBB;
	//Features<T> *feature;
	vector<Features<T> *> featuresindex;
	RTreeIndex RTroot[3];
	//MinBoundingBox<T> MBB;
	OGREnvelope ENCenvelope;
	int FeatureNum;
	int PointNum;
};

template<typename T>
struct QTreeLeafNode {
	Node* parent;
	vector<Features<T> *> featuresindex;
	RTreeIndex RTroot[3];
	OGREnvelope ENCenvelope;
	int FeatureNum;
	int PointNum;
};
int LayerType(const char *LayerName);

/*
template<typename ElemType>
class QuardTree
{
public:
	QuardTree();
	~QuardTree();
	void Insert(const Point<ElemType>& pos);
	void BalanceInsert(const Point<ElemType>& pos);
	int nodeCount();
	int TPLS();
	int Height();
	void RegionResearch(ElemType left, ElemType right, ElemType botom, ElemType top, int& visitednum, int& foundnum);
	void clear();
	void readpointdata(char *shapename, vector<Point<ElemType>> &Point_P);

private:
	Node<ElemType>* root;
	int Compare(const Node<ElemType>* node, const Point<ElemType>& pos);
	bool In_Region(Point<ElemType> t, ElemType left, ElemType right, ElemType botom, ElemType top);
	bool Rectangle_Overlapse_Region(ElemType L, ElemType R, ElemType B, ElemType T, ElemType left, ElemType right, ElemType botom, ElemType top);
	void RegionResearch(Node<ElemType>* t, ElemType left, ElemType right, ElemType botom, ElemType top, int& visitednum, int& foundnum);
	int Depth(Node<ElemType>* &);
	int nodeCount(const Node<ElemType>*);
	void clear(Node < ElemType>*& p);
	void Insert(Node<ElemType>*&, const Point<ElemType>& pos);//递归插入节点
};

//void readpointdata(char *shapename, vector<Point<float>> &Point_P);
void readlinedata(char *shapename, vector<Point<float>> &Point_L);
void readaredata(char *shapename, vector<Point<float>> &Point_A);
*/

class QuadTree
{
public:
	QuadTree() 
	{
		root = NULL;
	};
	~QuadTree() 
	{
		clear(root);
	};
	void QTreeResearch(OGREnvelope &SearchEnvelope, vector<Features<float> *> &result, vector<Node<float> *> &node);
	void InsertAdjust(Node<float>* temp, Features<float>* Feature);
	void Insert(Features<float> *Feature, int QT_k);
	void InsertLeaf(int &RT_k, bool opt);//插入R树
	void BalanceInsert(Features<float> *Feature);
	int nodeCount();
	int TPLS();
	int Height();
	void RegionResearch(float left, float right, float botom, float top, int& visitednum, int& foundnum);
	void clear();

	void RegionSearch(OGREnvelope &SearchEnvelope, vector<Features<float> *> &result, vector<RTreeNode *> &node, vector<Node<float> *> &Lnodes);//3DENCQRTree查询
	//int MaxDepth(Node<float>* node);
	void CreatQuad(OGREnvelope &envelope);
	void Change(int dir, OGREnvelope &old_ENCenvelope, OGREnvelope &new_ENCenvelope);//细分节点四至范围
	int Quadrant(Node<float>* node, Point<float>& pos);
	void readpointdata(const char *shapename, vector<Point<float>> &Point_P);
	void readlinedata(const char *shapename, vector<Point<float>> &Point_L);
	void readaredata(const char *shapename, vector<Point<float>> &Point_A);
	void readshape();
	//void FindData(char* lpPath, vector<Features<float>> &FeatureIndex);

private:
	Node<float>* root;
	int ComparePoint(const Node<float>* node, const Point<float>& pos);
	int Compare(const Node<float>* node, Features<float> *Feature);
	bool In_Region(Point<float> t, float left, float right, float botom, float top);
	bool Rectangle_Overlapse_Region(float L, float R, float B, float T, float left, float right, float botom, float top);
	void RegionResearch(Node<float>* t, float left, float right, float botom, float top, int& visitednum, int& foundnum);
	int Depth(Node<float>* &node);
	int nodeCount(const Node<float>* node);
	void clear(Node <float>*& p);
	void Insert(Node<float>*& p, Features<float> *Feature);//递归插入节点

	int MaxDepth(Node<float>* node);
	void InsertLeaf(Node<float>* node, int &RT_k, bool opt);//插入R树
	void InsertRTree(Node<float>* node, int &RT_k);//插入R树
	void RegionSearch(Node<float>* temp, OGREnvelope &SearchEnvelope, vector<Features<float> *> &result, vector<RTreeNode *> &node, vector<Node<float> *> &Lnodes);//3DENCQRTree查询
	
	void QTreeResearch(Node<float>* temp, OGREnvelope &SearchEnvelope, vector<Features<float> *> &result, vector<Node<float> *> &node);//四叉树查询
	int Envelope_In_Region(OGREnvelope &FeatureEnvelope, OGREnvelope &SearchEnvelope);
	bool Point_In_Region(Point<float> t, OGREnvelope &SearchEnvelope);
};



class Circle {
private:
	double r;
public:
	//构造函数 方法  按照不同的方式来创建对象
	//在成员方法中可以直接访问成员函数
	Circle(double ar) {
		r = ar;
	}
	Circle() {
		r = 0;
	}
	void setR(double ar) {
		r = ar;
	}
	double getR() {
		return r;
	}
	double len() {
		return 3.14*r * 2;
	}
	double area() {
		return 3.14*r*r;
	}
	//成员方法中直接访问成员属性　和　成员方法
	void show() {
		cout << "我是一个半径为:" << r << "的圆，面积为:" << area() << "，周长为:" << len() << endl;
	}
};