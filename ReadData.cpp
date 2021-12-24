#pragma once
//#include <afxdlgs.h>
#include <ReadData.h>
#include <rtreeindex.h>

float radius = 0.01f;        //半径
Point<float> CENTER = {118.0f, 0.0f, 39.0f};//场景中点
float Center_x = 118;             //场景中点x坐标
float Center_y = 39;             //场景中点y坐标
float Center_ratio = 0.5;         //场景比例
//QuardTree<float>* ENC_Tree = new QuardTree<float>();
int mm,nn;

int QuadTree::TPLS()
{
	cout<< Depth(root)<<endl;
	return Depth(root);
}

int QuadTree::ComparePoint(const Node<float>* node, const Point<float>& pos)
{
	Point<float> center;
	center.x = (node->ENCenvelope.MaxX + node->ENCenvelope.MinX) / 2;
	center.z = (node->ENCenvelope.MaxY + node->ENCenvelope.MinY) / 2;
	//center.x = (node->MBB.left_up.x + node->MBB.right_down.x) / 2;
	//center.z = (node->MBB.left_up.z + node->MBB.right_down.z) / 2 ;
	if (pos.x == center.x && pos.z == center.z) return 0;
	if (pos.x >= center.x && pos.z > center.z)  return 1;
	if (pos.x < center.x  && pos.z >= center.z) return 2;
	if (pos.x <= center.x && pos.z < center.z)  return 3;
	if (pos.x > center.x  && pos.z <= center.z) return 4;
	return -1;
}

int QuadTree::Compare(const Node<float>* node, Features<float> *Feature)
{
	int postion[4] = { 0,0,0,0 };
	for (int i = 0; i < Feature->point.size(); i++)
	{
		int ps = ComparePoint(node, Feature->point[i]);
		if (ps == 1) postion[0] += 1;
		if (ps == 2) postion[1] += 1;
		if (ps == 3) postion[2] += 1;
		if (ps == 4) postion[3] += 1;
	}

	if (postion[0] >= postion[1] && postion[0] >= postion[2] && postion[0] >= postion[3]) return 1;
	if (postion[1] > postion[0] && postion[1] >= postion[2] && postion[1] >= postion[3]) return 2;
	if (postion[2] > postion[0] && postion[2] > postion[1] && postion[2] >= postion[3]) return 3;
	if (postion[3] > postion[0] && postion[3] > postion[1] && postion[3] > postion[2]) return 4;

	return -1;
}

void QuadTree::BalanceInsert(Features<float> *Feature)
{
	Node<float>* node = (Node<float>*)malloc(sizeof(Node<float>));
	node->R[0] = NULL;
	node->R[1] = NULL;
	node->R[2] = NULL;
	node->R[3] = NULL;
	node->parent = NULL;
	//node->feature = Feature;
	if (root == NULL)
	{
		root = node;
		return;
	}
	Node<float>* temp = root;
	int direction = Compare(temp, Feature);
	if (direction == 0) return;
	while (temp->R[direction - 1] != NULL)
	{
		temp = temp->R[direction - 1];
		direction = Compare(temp, Feature);
		if (direction == 0) return;
	}
	temp->R[direction - 1] = node;
	node->parent = temp;

	Node<float>* tp = temp->parent;
	if (tp == NULL) return;
	int r = Compare(tp, temp->featuresindex[0]);

	if (abs(direction - r) == 2)
	{
		Node<float>* leaf = node;
		if (tp->R[abs(3 - r)] == NULL)
		{
			tp->R[r - 1] = NULL;
			temp->parent = leaf;
			leaf->R[r - 1] = temp;

			temp->R[abs(3 - r)] = NULL;
			Node<float>* Rt = tp->parent;
			if (Rt == NULL)
			{
				root = leaf;
				leaf->parent = NULL;

				leaf->R[abs(3 - r)] = tp;
				tp->parent = leaf;
				return;
			}
			tp->parent = NULL;
			int dd = Compare(Rt, tp->featuresindex[0]);

			Rt->R[dd - 1] = leaf;
			leaf->parent = Rt;

			leaf->R[abs(3 - r)] = tp;
			tp->parent = leaf;
		}
	}
}

void QuadTree::Insert(Node<float>*& p, Features<float> *Feature)
{
	if (p == NULL)
	{
		Node<float>* node = (Node<float>*)malloc(sizeof(Node<float>));
		node->R[0] = NULL;
		node->R[1] = NULL;
		node->R[2] = NULL;
		node->R[3] = NULL;
		//node->feature = Feature;
		p = node;
		return;
	}
	else
	{
		int d = Compare(p, Feature);
		if (d == 0) return;
		Insert(p->R[d - 1], Feature);
	}
}

void QuadTree::InsertAdjust(Node<float>* temp, Features<float>* Feature) 
{
	int direction = 0;
	Node<float>* node = new Node<float>;
	node->R[0] = NULL;
	node->R[1] = NULL;
	node->R[2] = NULL;
	node->R[3] = NULL;
	node->FeatureNum = 0;
	direction = Compare(temp, Feature);
	if (temp->R[direction - 1] == NULL)
	{
		node->featuresindex.push_back(Feature);
		node->PointNum = Feature->PointNum;
		node->FeatureNum++;
		Change(direction, temp->ENCenvelope, node->ENCenvelope);
		temp->R[direction - 1] = node;
		node->parent = temp;
	}
	else
	{
		temp->R[direction - 1]->featuresindex.push_back(Feature);
		temp->R[direction - 1]->PointNum += Feature->PointNum;
		temp->R[direction - 1]->FeatureNum++;
		node = temp->R[direction - 1];
		node->parent = temp;
	}
}

//插入要素
void QuadTree::Insert(Features<float>* Feature,int QT_k)
{
	int direction = 0;
	Node<float>* node = new Node<float>;
	node->R[0] = NULL;
	node->R[1] = NULL;
	node->R[2] = NULL;
	node->R[3] = NULL;
	node->FeatureNum = 0;
	direction = Compare(root, Feature);

	Node<float>* temp = root;
	while (temp->R[direction - 1] != NULL && temp->R[direction - 1]->FeatureNum >= QT_k) {
		temp = temp->R[direction - 1];
		for (int i = 0; i < temp->featuresindex.size(); i++)
		{
			InsertAdjust(temp,temp->featuresindex[i]);
		}
		temp->featuresindex.clear();
		direction = Compare(temp, Feature);
	}

	if (temp->R[direction - 1] == NULL)
	{
		node->featuresindex.push_back(Feature); 
		node->PointNum = Feature->PointNum;
		node->FeatureNum++;
		Change(direction,temp->ENCenvelope, node->ENCenvelope);
		temp->R[direction - 1] = node;
		temp->FeatureNum++;
		node->parent = temp;
	}
	else
	{
		temp->R[direction - 1]->featuresindex.push_back(Feature);
		temp->R[direction - 1]->PointNum += Feature->PointNum;
		temp->R[direction - 1]->FeatureNum++;
		temp->FeatureNum++;
		node = temp->R[direction - 1];
		node->parent = temp;
	}
}

//插入R树
void QuadTree::InsertLeaf(int &RT_k, bool opt)
{
	InsertLeaf(root, RT_k, opt);
}
//插入R树
void QuadTree::InsertLeaf(Node<float>* node, int &RT_k, bool opt)
{
	if (node->R[0] == NULL && node->R[1] == NULL && node->R[2] == NULL && node->R[3] == NULL)
	{
		if (opt==true)//分类插入
		{
			InsertRTree(node, RT_k);
			return;
		}
		else//不分类插入
		{
			for (int i = 0; i < node->featuresindex.size(); i++)
			{
				node->RTroot[0].Insert(node->featuresindex[i], node->ENCenvelope, RT_k);
			}
		}
	}

	for (int i = 0; i < 4; i++)
	{
		if (node->R[i] != NULL)
		{
			Node<float>* temp = node;
			temp = temp->R[i];
			InsertLeaf(temp, RT_k, opt);
		}
	}	
}

//插入R树
void QuadTree::InsertRTree(Node<float>* node, int &RT_k)
{
	//printf("节点要素数 %d \n", node->featuresindex.size());
	//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", node->ENCenvelope.MaxX, node->ENCenvelope.MaxY, node->ENCenvelope.MinX, node->ENCenvelope.MinY);

	for (int i = 0; i < node->featuresindex.size(); i++)
	{
		if (node->featuresindex[i]->FeatureType == 0)
		{
			node->RTroot[0].Insert(node->featuresindex[i], node->ENCenvelope, RT_k);
		}
		else if (node->featuresindex[i]->FeatureType == 1)
		{
			node->RTroot[1].Insert(node->featuresindex[i], node->ENCenvelope, RT_k);
		}
		else
		{
			node->RTroot[2].Insert(node->featuresindex[i], node->ENCenvelope, RT_k);
		}
	}
}

int QuadTree::nodeCount()
{
	return nodeCount(root);
}

int QuadTree::nodeCount(const Node<float>* node)
{
	if (node == NULL) return 0;
	return 1 + nodeCount(node->R[0]) + nodeCount(node->R[1]) + nodeCount(node->R[2]) + nodeCount(node->R[3]);
}

bool QuadTree::In_Region(Point<float> t, float left, float right, float botom, float top)
{
	return t.x >= left && t.x <= right && t.y >= botom && t.y <= top;
}

bool QuadTree::Rectangle_Overlapse_Region(float  L, float  R, float  B, float  T, float  left, float  right, float  botom, float  top)
{
	return L <= right && R >= left && B <= top && T >= botom;
	//return true;
}//优化查找速度

void QuadTree::RegionResearch(Node<float>* t, float left, float right, float botom, float top, int& visitednum, int& foundnum)
{
	if (t == NULL) return;
	float xc = t->pt.x;
	float zc = t->pt.z;
	if (In_Region(t->pt, left, right, botom, top)) { ++foundnum; }
	if (t->R[0] != NULL && Rectangle_Overlapse_Region(xc, right, zc, top, left, right, botom, top))
	{
		visitednum++;
		RegionResearch(t->R[0], xc>left ? xc : left, right, zc>botom ? zc : botom, top, visitednum, foundnum);
	}
	if (t->R[1] != NULL && Rectangle_Overlapse_Region(left, xc, zc, top, left, right, botom, top))
	{
		visitednum++;
		RegionResearch(t->R[1], left, xc>right ? right : xc, zc>botom ? zc : botom, top, visitednum, foundnum);
	}
	if (t->R[2] != NULL && Rectangle_Overlapse_Region(left, xc, botom, zc, left, right, botom, top))
	{
		visitednum++;
		RegionResearch(t->R[2], left, xc<right ? xc : right, botom, zc<top ? zc : top, visitednum, foundnum);
	}
	if (t->R[3] != NULL && Rectangle_Overlapse_Region(xc, right, botom, zc, left, right, botom, top))
	{
		visitednum++;
		RegionResearch(t->R[3], xc>left ? xc : left, right, botom, zc<top ? zc : top, visitednum, foundnum);
	}
}

void QuadTree::clear()
{
	clear(root);
}

void QuadTree::clear(Node<float>* &p)
{
	if (p == NULL) return;
	if (p->R[0]) clear(p->R[0]);
	if (p->R[1]) clear(p->R[1]);
	if (p->R[2]) clear(p->R[2]);
	if (p->R[3]) clear(p->R[3]);
	free(p);
	p = NULL;
}

void QuadTree::RegionResearch(float left, float right, float botom, float top, int& visitednum, int& foundnum)
{
	RegionResearch(root, left, right, botom, top, visitednum, foundnum);
}

int QuadTree::Envelope_In_Region(OGREnvelope &FeatureEnvelope, OGREnvelope &SearchEnvelope)
{
	if (FeatureEnvelope.MaxX<SearchEnvelope.MaxX && FeatureEnvelope.MinX>SearchEnvelope.MinX && FeatureEnvelope.MaxY<SearchEnvelope.MaxY && FeatureEnvelope.MinY>SearchEnvelope.MinY)
	{
		return 1;
	}
	double minx = MAX(FeatureEnvelope.MinX, SearchEnvelope.MinX);
	double miny = MAX(FeatureEnvelope.MinY, SearchEnvelope.MinY);
	double maxx = MIN(FeatureEnvelope.MaxX, SearchEnvelope.MaxX);
	double maxy = MIN(FeatureEnvelope.MaxY, SearchEnvelope.MaxY);
	if (minx > maxx || miny > maxy)
	{
		return 0;
	}
	else
	{
		return 2;
	}
}
bool QuadTree::Point_In_Region(Point<float> t, OGREnvelope &SearchEnvelope)
{
	return t.x >= SearchEnvelope.MinX && t.x <= SearchEnvelope.MaxX && t.z >= SearchEnvelope.MinY && t.z <= SearchEnvelope.MaxY;
}


//四叉树查询
void QuadTree::QTreeResearch(OGREnvelope &SearchEnvelope, vector<Features<float> *> &result,vector<Node<float> *> &node)
{
	if (root == NULL) return;
	QTreeResearch(root, SearchEnvelope, result, node);
	//printf("nn %d \n", nn);
	
}
//四叉树查询
void QuadTree::QTreeResearch(Node<float>* temp, OGREnvelope &SearchEnvelope, vector<Features<float> *> &result, vector<Node<float> *> &node)
{

	if (temp->R[0] == NULL && temp->R[1] == NULL && temp->R[2] == NULL && temp->R[3] == NULL)
	{
		node.push_back(temp);
		for (int i = 0; i < temp->featuresindex.size(); i++)
		{
			/*
			if (Envelope_In_Region(node->featuresindex[i]->ENCenvelope, SearchEnvelope) && Envelope_In_Region(node->ENCenvelope, SearchEnvelope) ==false)
			{
				printf("node: MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", node->ENCenvelope.MaxX, node->ENCenvelope.MaxY, node->ENCenvelope.MinX, node->ENCenvelope.MinY);
				printf("feature:MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", node->featuresindex[i]->ENCenvelope.MaxX, node->featuresindex[i]->ENCenvelope.MaxY, node->featuresindex[i]->ENCenvelope.MinX, node->featuresindex[i]->ENCenvelope.MinY);

			}
		*/
			if (temp->featuresindex[i]->ENCenvelope.Intersects(SearchEnvelope))
			{
				//printf("P %d \n", temp->featuresindex[i]->point.size());
				Features<float>* FeatureTemp;
				FeatureTemp = temp->featuresindex[i];
				//FeatureTemp->point.clear();
				//printf("P %d \n", temp->featuresindex[i]->point.size());
				for (int j = 0; j < temp->featuresindex[i]->point.size(); j++)
				{
					if (Point_In_Region(temp->featuresindex[i]->point[j], SearchEnvelope))
					{
						//FeatureTemp->point.push_back(temp->featuresindex[i]->point[j]);
						int m;
						m++;
					}
				}
				result.push_back(FeatureTemp);			
			}
		}
	}

	for (int i = 0; i < 4; i++)
	{
		if (temp->R[i] != NULL)
		{
			QTreeResearch(temp->R[i], SearchEnvelope, result, node);
		}
	}
}


//3DENCQRTree查询
void QuadTree::RegionSearch(OGREnvelope &SearchEnvelope, vector<Features<float> *> &result, vector<RTreeNode *> &node, vector<Node<float> *> &Lnodes)
{
	if (root == NULL) return;
	RegionSearch(root, SearchEnvelope, result, node, Lnodes);
	//printf("mm %d \n", mm);
	
}
//3DENCQRTree查询
void QuadTree::RegionSearch(Node<float>* temp, OGREnvelope &SearchEnvelope, vector<Features<float> *> &result, vector<RTreeNode *> &node, vector<Node<float> *> &Lnodes)
{
	if (temp->R[0] == NULL && temp->R[1] == NULL && temp->R[2] == NULL && temp->R[3] == NULL)
	{
		if (Envelope_In_Region(temp->ENCenvelope, SearchEnvelope) == 1)//包含
		{
			Lnodes.push_back(temp);
			result.insert(result.end(), temp->featuresindex.begin(), temp->featuresindex.end());
			//temp->RTroot[0].root->GetNode(node);
		}
		else if (Envelope_In_Region(temp->ENCenvelope, SearchEnvelope) == 2)//不包含
		{/**/
			Lnodes.push_back(temp);
			//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", temp->ENCenvelope.MaxX, temp->ENCenvelope.MaxY, temp->ENCenvelope.MinX, temp->ENCenvelope.MinY);
			//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", SearchEnvelope.MaxX, SearchEnvelope.MaxY, SearchEnvelope.MinX, SearchEnvelope.MinY);
			temp->RTroot[0].root->QRTreeSearch(SearchEnvelope, result, node);
			temp->RTroot[1].root->QRTreeSearch(SearchEnvelope, result, node);
			temp->RTroot[2].root->QRTreeSearch(SearchEnvelope, result, node);
		}
		else//相交
		{
			Lnodes.push_back(temp);
			temp->RTroot[0].root->QRTreeSearch(SearchEnvelope, result, node);
			temp->RTroot[1].root->QRTreeSearch(SearchEnvelope, result, node);
			temp->RTroot[2].root->QRTreeSearch(SearchEnvelope, result, node);
			//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", temp->ENCenvelope.MaxX, temp->ENCenvelope.MaxY, temp->ENCenvelope.MinX, temp->ENCenvelope.MinY);

		}
		/*Lnodes.push_back(temp);
		temp->RTroot[0].root->QRTreeSearch(SearchEnvelope, result, node);
		temp->RTroot[1].root->QRTreeSearch(SearchEnvelope, result, node);
		temp->RTroot[2].root->QRTreeSearch(SearchEnvelope, result, node);*/
	}

	for (int i = 0; i < 4; i++)
	{
		if (temp->R[i] != NULL )
		{
			RegionSearch(temp->R[i], SearchEnvelope, result, node, Lnodes);
		}
	}
}


int QuadTree::Depth(Node<float>* &node)
{
	if (node == NULL) return 0;
	int dep = 0;
	Node<float>* tp = root;
	while (tp->pt.x != node->pt.x || tp->pt.z != node->pt.z)
	{
		dep++;
		tp = tp->R[Compare(tp, node->featuresindex[0]) - 1];
		if (tp == NULL) break;
	}
	return dep + Depth(node->R[0]) + Depth(node->R[1]) + Depth(node->R[2]) + Depth(node->R[3]);
}

int QuadTree::MaxDepth(Node<float>* node)
{
	if (node == NULL) return 0;
	int dep = 0;
	Node<float>* tp = root;
	while (tp->pt.x != node->pt.x || tp->pt.z != node->pt.z)
	{
		dep++;
		tp = tp->R[Compare(tp, node->featuresindex[0]) - 1];
		if (tp == NULL) break;
	}
	dep = max(dep, MaxDepth(node->R[0]));
	dep = max(dep, MaxDepth(node->R[1]));
	dep = max(dep, MaxDepth(node->R[2]));
	dep = max(dep, MaxDepth(node->R[3]));
	return dep;
}

int QuadTree::Height()
{
	return MaxDepth(root);
}

int QuadTree::Quadrant(Node<float>* node, Point<float>& pos)
{
	if (pos.x == node->pt.x && pos.z == node->pt.z) return 0;
	if (pos.x >= node->pt.x && pos.z>node->pt.z)  return 1;
	if (pos.x<node->pt.x  && pos.z >= node->pt.z) return 2;
	if (pos.x <= node->pt.x && pos.z<node->pt.z)  return 3;
	if (pos.x>node->pt.x  && pos.z <= node->pt.z) return 4;
	return -1;
}

void QuadTree::CreatQuad(OGREnvelope &envelope)
{
	Node<float>* node = (Node<float>*)malloc(sizeof(Node<float>));
	node->R[0] = NULL;
	node->R[1] = NULL;
	node->R[2] = NULL;
	node->R[3] = NULL;
	node->PointNum = 0;
	node->FeatureNum = 0;
	node->ENCenvelope = envelope;
	node->featuresindex.clear();
	if (root == NULL)
	{
		root = node;
		return;
	}
	else
	{
		printf("根节点已存在！ %n\n");
	}
}

void QuadTree::Change(int dir, OGREnvelope &old_ENCenvelope, OGREnvelope &new_ENCenvelope)//包围盒变换
{
	OGREnvelope envelopetemp;
	if (dir == 0)
	{
		envelopetemp = old_ENCenvelope;
		new_ENCenvelope = envelopetemp;
	}
	if (dir == 1)
	{
		envelopetemp.MaxX = old_ENCenvelope.MaxX;
		envelopetemp.MinX = (old_ENCenvelope.MaxX + old_ENCenvelope.MinX) / 2;
		envelopetemp.MaxY = old_ENCenvelope.MaxY;
		envelopetemp.MinY = (old_ENCenvelope.MaxY + old_ENCenvelope.MinY) / 2;
		new_ENCenvelope = envelopetemp;
	}
	if (dir == 2)
	{
		envelopetemp.MaxX = (old_ENCenvelope.MaxX + old_ENCenvelope.MinX) / 2;
		envelopetemp.MinX = old_ENCenvelope.MinX;
		envelopetemp.MaxY = old_ENCenvelope.MaxY;
		envelopetemp.MinY = (old_ENCenvelope.MaxY + old_ENCenvelope.MinY) / 2;
		new_ENCenvelope = envelopetemp;
	}	
	if (dir == 3)
	{
		envelopetemp.MaxX = (old_ENCenvelope.MaxX + old_ENCenvelope.MinX) / 2;
		envelopetemp.MinX = old_ENCenvelope.MinX;
		envelopetemp.MaxY = (old_ENCenvelope.MaxY + old_ENCenvelope.MinY) / 2;
		envelopetemp.MinY = old_ENCenvelope.MinY;
		new_ENCenvelope = envelopetemp;
	}
	if (dir == 4)
	{
		envelopetemp.MaxX = old_ENCenvelope.MaxX;
		envelopetemp.MinX = (old_ENCenvelope.MaxX + old_ENCenvelope.MinX) / 2;
		envelopetemp.MaxY = (old_ENCenvelope.MaxY + old_ENCenvelope.MinY) / 2;
		envelopetemp.MinY = old_ENCenvelope.MinY;
		new_ENCenvelope = envelopetemp;
	}
}

/*
//加载点数据
void QuadTree::readSOUNDGdata(const char *shapename, vector<Point<float>> &Point_P)
{
	//ifstream inFile("C:\\Users\\Administrator\\Desktop\\test.shp", ios::binary | ios::in);
		
	
	float num, lat, lon, sea;
	int data_row = 1;//行数
	int data_col = 1;//列数
	int data_num = 1;//数据数量
	int count = 0;  //计数器，记录已读出的浮点数

	if ((fp = fopen(shapename, "rb")) == NULL)
	{
		printf("请确认文件(%s)是否存在!\n", shapename);
		exit(1);
	}
	else
	{
		printf("文件(%s)存在!\n", shapename);
	}
	while (!feof(fp))
	{
		char flag = fgetc(fp);
		if (flag == '\n')
			data_row++;

		if (flag == ',' || flag == '\n' || flag == ' ')
		{
			data_num++;
			data_col = data_num / data_row;
		}

	}
	printf("数据行数为：%d\n", data_row);
	printf("数据列数为：%d\n", data_col);
	printf("数据数量为：%d\n", data_num);

	float **pointdata = (float **)malloc(sizeof(float *) * data_row);
	for (int i = 0; i<data_row; i++) {
		pointdata[i] = (float*)malloc(sizeof(float *) * data_col);
	}
	//	int *index = (int*)calloc(data_row, sizeof(int));

	float lat_max, lat_min, lon_max, lon_min;
	int number[4];

	//fp = fopen("datatest.txt", "rb");    //打开datatest.txt
	fp = fopen(shapename, "rb");    //打开SOUNDG_P.txt
	while (!feof(fp))
	{
		//fscanf(fp, "%f %f %f", &lat, &lon, &sea);   //datatest.txt
		fscanf(fp, "%f,%f,%f", &sea, &lon, &lat);    //SOUNDG_P.txt
		if (num == NULL)
		{
			break;
		}
		pointdata[count][0] = count + 1;
		pointdata[count][1] = lat;
		pointdata[count][2] = lon;
		pointdata[count][3] = sea;
		//printf("%f %f %f %f", num, lat, lon, sea);
		count++;

		if (count < 2)
		{
			lat_max = lat_min = lat;
			lon_max = lon_min = lon;
			for (int i = 0; i < 4; i++)
			{
				number[i] = 1;
			}
		}
		if (lat > lat_max)
		{
			lat_max = lat;
			number[0] = count;
		}
		if (lat < lat_min)
		{
			lat_min = lat;
			number[1] = count;
		}
		if (lon > lon_max)
		{
			lon_max = lon;
			number[2] = count;
		}
		if (lon < lon_min)
		{
			lon_min = lon;
			number[3] = count;
		}
	}

	Center_x = (lon_max + lon_min) / 2;
	Center_y = (lat_max + lat_min) / 2;
	Center_ratio = Center_x - lon_min;

	float Point_x, Point_y, Point_z;
	for (int i = 0; i < data_row; i++) //归一化、删除多余点
	{
		Point_x = (pointdata[i][2] - Center_x) / Center_ratio;
		Point_y = -(pointdata[i][1] - Center_y) / Center_ratio;
		Point_z = -0.01f*pointdata[i][3];

		//sounding.push_back(Point(100 * (seadata[i][1] - 38.89f), -0.01f*seadata[i][3], 100 * (seadata[i][2] - 117.69f)));           //datatest.txt
		if (i > 0 && pointdata[i][1] == pointdata[i - 1][1] && pointdata[i][2] == pointdata[i - 1][2]) {}
		else
		{
			Point_P.push_back(Point<float>(Point_x, Point_z, Point_y));           //SOUNDG_P.txt
			Insert(Point_P[i]);
		}
			
	}
	//printf("%d\n", depth(root));
	
	printf("最大纬度 %f,%d\n", lat_max, number[0]);     //最大纬度
	printf("最小纬度 %f,%d\n", lat_min, number[1]);     //最小纬度
	printf("最大经度 %f,%d\n", lon_max, number[2]);     //最大经度
	printf("最小经度 %f,%d\n", lon_min, number[3]);     //最小经度

	fclose(fp);
}*/
/**/
//加载点数据
void QuadTree::readpointdata(const char *shapename, vector<Point<float>> &Point_P)
{
	//ifstream inFile("C:\\Users\\Administrator\\Desktop\\test.shp", ios::binary | ios::in);

	FILE *fp;            /*文件指针*/
	float num, lat, lon, sea;
	int data_row = 0;//行数
	int data_col = 2;//列数
	int data_num = 0;//数据数量
	int count = 0;  //计数器，记录已读出的浮点数

	if ((fp = fopen(shapename, "rb")) == NULL)
	{
		printf("请确认文件(%s)是否存在!\n", shapename);
		exit(1);
	}
	else
	{
		printf("文件(%s)存在!\n", shapename);
	}
	while (!feof(fp))
	{
		char flag = fgetc(fp);
		if (flag == '\n')
			data_row++;
		/*
		if (flag == ',' || flag == '\n' || flag == ' ')
		{
			data_num++;
			
		}
		fscanf(fp, "%f %f", &lon, &lat);

		if (lon == NULL)
		{
			break;
		}
		else
		{
			data_row++;
			data_num += 3;
		}
		data_col = data_num / data_row;
		//printf("数据行数为：%f\n", lon);*/
	}
	
	float **pointdata = (float **)malloc(sizeof(float *) * data_row);
	for (int i = 0; i<data_row; i++) {
		pointdata[i] = (float*)malloc(sizeof(float *) * data_col);
	}
	//	int *index = (int*)calloc(data_row, sizeof(int));

	float lat_max, lat_min, lon_max, lon_min;
	int number[4];

	//fp = fopen("datatest.txt", "rb");    //打开datatest.txt
	fp = fopen(shapename, "rb");    //打开SOUNDG_P.txt
	while (!feof(fp))
	{
		//fscanf(fp, "%f %f %f", &lat, &lon, &sea);   //datatest.txt

		fscanf(fp, "%f %f", &lon, &lat);    //SOUNDG_P.txt
		char flag = fgetc(fp);
		if (flag == -1)
		{
			break;
		}
		pointdata[count][0] = count + 1;
		pointdata[count][1] = lat;
		pointdata[count][2] = lon;
		pointdata[count][3] = 0;
		//printf("%f %f %f %f", num, lat, lon, sea);
		count++;

		if (count < 2)
		{
			lat_max = lat_min = lat;
			lon_max = lon_min = lon;
			for (int i = 0; i < 4; i++)
			{
				number[i] = 1;
			}
		}
		if (lat > lat_max)
		{
			lat_max = lat;
			number[0] = count;
		}
		if (lat < lat_min)
		{
			lat_min = lat;
			number[1] = count;
		}
		if (lon > lon_max)
		{
			lon_max = lon;
			number[2] = count;
		}
		if (lon < lon_min)
		{
			lon_min = lon;
			number[3] = count;
		}
	}

/*	Center_x = (lon_max + lon_min) / 2;
	Center_y = (lat_max + lat_min) / 2;
	Center_ratio = Center_x - lon_min;
	*/
	float Point_x, Point_y, Point_z;
	for (int i = 0; i < data_row; i++) //归一化、删除多余点
	{
		Point_x = (pointdata[i][2] - Center_x) / Center_ratio;
		Point_y = -(pointdata[i][1] - Center_y) / Center_ratio;//z轴取负数，将数据坐标翻转，匹配视坐标系
		Point_z = pointdata[i][3];

		//sounding.push_back(Point(100 * (seadata[i][1] - 38.89f), -0.01f*seadata[i][3], 100 * (seadata[i][2] - 117.69f)));           //datatest.txt
		if (i > 0 && pointdata[i][1] == pointdata[i - 1][1] && pointdata[i][2] == pointdata[i - 1][2]) {}
		else
		{
			Point_P.push_back(Point<float>(Point_x, Point_z, Point_y));           //SOUNDG_P.txt
			//Insert(Point_P[i]);
		}
	}

	//data_num = (data_row - 1) * data_col;
	printf("数据行数为：%d\n", data_row);
	printf("数据列数为：%d\n", data_col);
	printf("数据数量为：%d\n", count);
	printf("有效数据总数为：%d\n", Point_P.size());
	printf("最大纬度 %f,%d\n", lat_max, number[0]);     //最大纬度
	printf("最小纬度 %f,%d\n", lat_min, number[1]);     //最小纬度
	printf("最大经度 %f,%d\n", lon_max, number[2]);     //最大经度
	printf("最小经度 %f,%d\n", lon_min, number[3]);     //最小经度

	fclose(fp);
}

//加载线数据
void QuadTree::readlinedata(const char *shapename, vector<Point<float>> &Point_L)
{
	FILE *fp;            /*文件指针*/
	int data_row = 0;//行数
	int data_col = 3;//列数
	int data_num = 0;//数据数量
	int count = 0;  //计数器，记录已读出的浮点数
	float num, lat, lon, lat_max, lat_min, lon_max, lon_min;

	if ((fp = fopen(shapename, "rb")) == NULL)
	{
		printf("请确认文件(%s)是否存在!\n", shapename);
		exit(1);
	}
	else
	{
		printf("文件(%s)存在!\n", shapename);
	}
	while (!feof(fp))
	{
		char flag = fgetc(fp);
		if (flag == '\n')
			data_row++;
		/*
		if (flag == ',' || flag == '\n' || flag == ' ')
		{
			data_num++;
			data_col = data_num / data_row;
		}
*/
	}

	float **linedata = (float **)malloc(sizeof(float *) * data_row);
	for (int i = 0; i<data_row; i++) {
		linedata[i] = (float*)malloc(sizeof(float *) * data_col);
	}

	int number[4];

	fp = fopen(shapename, "rb");
	while (!feof(fp))
	{
		fscanf(fp, "%f %f %f", &num, &lon, &lat);
		char flag = fgetc(fp);
		if (flag == -1)
		{
			break;
		}
		linedata[count][0] = count + 1;
		linedata[count][1] = lat;
		linedata[count][2] = lon;
		linedata[count][3] = num;

		count++;

		if (count < 2)
		{
			lat_max = lat_min = lat;
			lon_max = lon_min = lon;
			for (int i = 0; i < 4; i++)
			{
				number[i] = 1;
			}
		}
		if (lat > lat_max)
		{
			lat_max = lat;
			number[0] = count;
		}
		if (lat < lat_min)
		{
			lat_min = lat;
			number[1] = count;
		}
		if (lon > lon_max)
		{
			lon_max = lon;
			number[2] = count;
		}
		if (lon < lon_min)
		{
			lon_min = lon;
			number[3] = count;
		}
	}
	for (int i = 0; i < data_row; i++) //归一化,去掉重复和无效数据点
	{
		float Point_x = (linedata[i][2] - Center_x) / Center_ratio;
		float Point_y = -(linedata[i][1] - Center_y) / Center_ratio;//z轴取负数，将数据坐标翻转，匹配视坐标系
		float Point_z = linedata[i][3];

		if (i > 0 && linedata[i][3] == linedata[i - 1][3] && linedata[i][1] == linedata[i - 1][1] && linedata[i][2] == linedata[i - 1][2]) {}
		if (i > 1 && linedata[i][3] == linedata[i - 1][3] && linedata[i - 1][3] == linedata[i - 2][3])
		{
			float temp = (linedata[i][1] - linedata[i - 1][1]) / (linedata[i - 1][1] - linedata[i - 2][1]) - (linedata[i][2] - linedata[i - 1][2]) / (linedata[i - 1][2] - linedata[i - 2][2]);
			if (temp > -0.1 && temp < 0.1)
			{
				Point_L.pop_back();
				Point_L.push_back(Point<float>(Point_x, Point_z, Point_y));
			}
			else 
			{
				Point_L.push_back(Point<float>(Point_x, Point_z, Point_y));
			}
		}
		else
		{
			Point_L.push_back(Point<float>(Point_x, Point_z, Point_y));
		}
		//printf("%d,%f\n", i, linedata[i][2]);
	}

	//data_num = (data_row - 1) * data_col;
	printf("数据行数为：%d\n", data_row);
	printf("数据列数为：%d\n", data_col);
	printf("数据数量为：%d\n", count);
	printf("有效数据总数为：%d\n", Point_L.size());
	printf("最大纬度 %f,%d\n", lat_max, number[0]);     //最大纬度
	printf("最小纬度 %f,%d\n", lat_min, number[1]);     //最小纬度
	printf("最大经度 %f,%d\n", lon_max, number[2]);     //最大经度
	printf("最小经度 %f,%d\n", lon_min, number[3]);     //最小经度

	fclose(fp);
}

//加载面数据
void QuadTree::readaredata(const char *shapename, vector<Point<float>> &Point_A)
{
	FILE *fp;            /*文件指针*/
	int data_row = 0;//行数
	int data_col = 3;//列数
	int data_num = 0;//数据数量
	int count = 0;  //计数器，记录已读出的浮点数
	float num, lat, lon, lat_max, lat_min, lon_max, lon_min;

	if ((fp = fopen(shapename, "rb")) == NULL)
	{
		printf("请确认文件(%s)是否存在!\n", shapename);
		exit(1);
	}
	else
	{
		printf("文件(%s)存在!\n", shapename);
	}
	while (!feof(fp))
	{
		char flag = fgetc(fp);
		if (flag == '\n')
			data_row++;
/*
		if (flag == ',' || flag == '\n' || flag == ' ')
		{
			data_num++;
			
		}
		fscanf(fp, "%f %f %f", &num, &lon, &lat);
		
		if (num != 0 && num == NULL)
		{
			break;
		}
		else
		{
			data_row++;
			data_num += 3;
		}
		data_col = data_num / data_row;*/
	}
	
	float **aredata = (float **)malloc(sizeof(float *) * data_row);
	for (int i = 0; i<data_row; i++) {
		aredata[i] = (float*)malloc(sizeof(float *) * data_col);
	}

	int number[4];

	fp = fopen(shapename, "rb");
	while (!feof(fp))
	{
		fscanf(fp, "%f %f %f", &num, &lon, &lat);
		char flag = fgetc(fp);
		if (flag == -1)
		{
			break;
		}
		aredata[count][0] = count + 1;
		aredata[count][1] = lat;
		aredata[count][2] = lon;
		aredata[count][3] = num;
		//printf("%f %f %f", num, lat, lon);
		count++;

		if (count < 2)
		{
			lat_max = lat_min = lat;
			lon_max = lon_min = lon;
			for (int i = 0; i < 4; i++)
			{
				number[i] = 1;
			}
		}
		if (lat > lat_max)
		{
			lat_max = lat;
			number[0] = count;
		}
		if (lat < lat_min)
		{
			lat_min = lat;
			number[1] = count;
		}
		if (lon > lon_max)
		{
			lon_max = lon;
			number[2] = count;
		}
		if (lon < lon_min)
		{
			lon_min = lon;
			number[3] = count;
		}
	}
	for (int i = 0; i < data_row; i++) //归一化、去掉重复和无效数据点
	{
		float Point_x = (aredata[i][2] - Center_x) / Center_ratio;
		float Point_y = -(aredata[i][1] - Center_y) / Center_ratio;//z轴取负数，将数据坐标翻转，匹配视坐标系
		float Point_z = aredata[i][3];
		if (i > 0 && aredata[i][1] == aredata[i - 1][1] && aredata[i][2] == aredata[i - 1][2]) {}
		if (i > 1 && aredata[i][3] == aredata[i - 1][3] && aredata[i - 1][3] == aredata[i - 2][3])
		{
			float temp = (aredata[i][1] - aredata[i - 1][1]) / (aredata[i - 1][1] - aredata[i - 2][1]) - (aredata[i][2] - aredata[i - 1][2]) / (aredata[i - 1][2] - aredata[i - 2][2]);
			if (aredata[i][1] == aredata[i - 1][1] && aredata[i - 1][1] == aredata[i - 2][1])
			{
				Point_A.pop_back();
				Point_A.push_back(Point<float>(Point_x, Point_z, Point_y));
				//
			}
			if (aredata[i][2] == aredata[i - 1][2] && aredata[i - 1][2] == aredata[i - 2][2])
			{
				Point_A.pop_back();
				Point_A.push_back(Point<float>(Point_x, Point_z, Point_y));
				//printf("%d\n", i);
			}
			if (temp > -0.1 && temp < 0.1)
			{
				Point_A.pop_back();
				Point_A.push_back(Point<float>(Point_x, Point_z, Point_y));
			}
			else
			{
				Point_A.push_back(Point<float>(Point_x, Point_z, Point_y));
			}
		}
		else
		{
			Point_A.push_back(Point<float>(Point_x, Point_z, Point_y));
		}
		//printf("%d,%d\n", i,Point_A.size());
	}

	//data_num = (data_row - 1) * data_col;
	printf("数据行数为：%d\n", data_row);
	printf("数据列数为：%d\n", data_col);
	printf("数据数量为：%d\n", count);
	printf("有效数据总数为：%d\n", Point_A.size());
	printf("最大纬度 %f,%d\n", lat_max, number[0]);     //最大纬度
	printf("最小纬度 %f,%d\n", lat_min, number[1]);     //最小纬度
	printf("最大经度 %f,%d\n", lon_max, number[2]);     //最大经度
	printf("最小经度 %f,%d\n", lon_min, number[3]);     //最小经度
	fclose(fp);
}

//读shape文件
void QuadTree::readshape() 
{
	GDALAllRegister();
	GDALDataset   *poDS;
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");  //解决中文乱码问题
											   //读取shp文件
	poDS = (GDALDataset*)GDALOpenEx("E:/ArcGIS/CN323101/RESARE_A.shp", GDAL_OF_VECTOR, NULL, NULL, NULL);

	if (poDS == NULL)
	{
		printf("Open failed.\n%s");
	}
	printf("Open successful.\n%s");
	OGRLayer  *poLayer;
	poLayer = poDS->GetLayer(0); //读取层
	OGRFeature *poFeature;

	poLayer->ResetReading();
	int i = 0;
	while ((poFeature = poLayer->GetNextFeature()) != NULL)
	{
 		if (poFeature->GetFieldAsDouble("AREA")<1) continue; //去掉面积过小的polygon
		i = i++;
		printf("%d" ,i);
		//cout << i << "  ";
		OGRFeatureDefn *poFDefn = poLayer->GetLayerDefn();
		int iField;
		int n = poFDefn->GetFieldCount(); //获得字段的数目，不包括前两个字段（FID,Shape);
		printf("%d\n", n);
		//cout << n << endl;
		for (iField = 0; iField <n; iField++)
		{
			//输出每个字段的值
			cout << poFeature->GetFieldAsString(iField) << "    ";
		}
		cout << endl;
		OGRFeature::DestroyFeature(poFeature);
	}
	GDALClose(poDS);
	system("pause");
/*	*/
}
/*
//void QuadTree::FindData(char* lpPath, vector<string> &fileList)
void QuadTree::FindData(char* lpPath, vector<Features<float>> &FeatureIndex)
{
	char szFind[MAX_PATH];
	WIN32_FIND_DATA FindFileData;

	strcpy(szFind, lpPath);
	strcat(szFind, "\\*.txt");

	HANDLE hFind = ::FindFirstFile(szFind, &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)  return;

	while (true)
	{
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)//判断查找的是不是文件夹
		{

			if (FindFileData.cFileName[0] != '.')
			{
				char szFile[MAX_PATH];
				strcpy(szFile, lpPath);
				strcat(szFile, "\\");
				strcat(szFile, (char*)(FindFileData.cFileName));
				FindData(szFile, FeatureIndex);
			}
		}
		else
		{
			//std::cout << FindFileData.cFileName << std::endl; 
			Features<float> Feature;
			string str = FindFileData.cFileName;
			const char *shapename = str.data();
			//vector<Point<float>> Point, Line, Are;
			Feature.featureID = shapename;

			for (int j = 0; shapename[j] != '\0'; j++)
			{
				char shapedataPath[100];
				strcpy(shapedataPath, lpPath);
				strcat(shapedataPath, shapename);

				if (shapename[j] == '_' && shapename[j + 1] == 'P')
				{
					Feature.shapetype = 1;
					readpointdata(shapedataPath, Feature.point, Feature.MBB);
					//Feature.point = Point;
					Feature.shapenumber = Feature.point.size();
					//printf("数据量为：%d\n", Feature.shapenumber);
					FeatureIndex.push_back(Feature);
					break;
				}
				if (shapename[j] == '_' && shapename[j + 1] == 'L')
				{
					Feature.shapetype = 3;
					readlinedata(shapedataPath, Feature.point, Feature.MBB);
					//Feature.point = Line;
					Feature.shapenumber = Feature.point.size();
					//printf("数据量为：%d\n", Feature.shapenumber);
					FeatureIndex.push_back(Feature);
					break;
				}
				if (shapename[j] == '_' && shapename[j + 1] == 'A')
				{
					Feature.shapetype = 5;
					readaredata(shapedataPath, Feature.point, Feature.MBB);
					//Feature.point = Are;
					Feature.shapenumber = Feature.point.size();
					//printf("数据量为：%d\n", Feature.shapenumber);
					FeatureIndex.push_back(Feature);
					break;
				}
				else
				{
					continue;
				}
			}
		}
		if (!FindNextFile(hFind, &FindFileData))  break;
	}

	FindClose(hFind);
}
*/
//判断要素分类类型
int LayerType(const char *LayerName)
{
	const char* lpPath = "E:\\program\\oceanFFT\\data\\ENCFeature.txt";
	FILE *fp;            /*文件指针*/

	if ((fp = fopen(lpPath, "rb")) == NULL)
	{
		printf("请确认文件(%s)是否存在!\n", lpPath);
		exit(1);
	}
	else
	{
		//printf("文件(%s)存在!\n", lpPath);
	}
	int Num;
	string str;

	while (!feof(fp))
	{
		fscanf(fp, "%s %d", &str, &Num);
		const char *Name = str.data();
		if (strcmp(Name, LayerName) == 0)
		{
			fclose(fp);
			return Num;
			break;
		}
	}
	fclose(fp);
	return -1;
}

/*
void QuadTree::Insert(Features<float> *Feature)
{
	//printf("pos %f %f %f\n", pos.x, pos.y, pos.z);
	int direction, len = 0;
	Node<float>* node = (Node<float>*)malloc(sizeof(Node<float>));
	node->R[0] = NULL;
	node->R[1] = NULL;
	node->R[2] = NULL;
	node->R[3] = NULL;
	node->feature = Feature;

	direction = Compare(root, Feature);

	Node<float>* temp = root;
	if (direction == 0) return;//节点已存在 
	len = 1;
	while (temp->R[direction - 1] != NULL)
	{
		OGREnvelope *old_ENCenvelope = temp->ENCenvelope;
		temp = temp->R[direction - 1];
		Change(direction, old_ENCenvelope, temp->ENCenvelope);
		direction = Compare(temp, Feature);
		if (direction == 0) return;

	}
	node->ENCenvelope = temp->ENCenvelope;
	temp->R[direction - 1] = node;
	node->parent = temp;
	//printf("位置 %d\n", direction);
	//printf("MBB %f %f %f %f\n", temp->MBB.left_up.x, temp->MBB.left_up.z, temp->MBB.right_down.x, temp->MBB.right_down.z);
	//Insert(root, pos);//递归插入节点
}
*/


/*
void ReadShapeData()
{
	//注册所有的文件格式驱动
	OGRRegisterAll();

	//打开point.shp文件
	OGRDataSource *poDS = OGRSFDriverRegistrar::Open("point.shp", FALSE);

	//获取点层
	OGRLayer *poLayer = poDS->GetLayerByName("point");

	OGRFeature *poFeature;

	//重置该层，确保从层的开始读取数据
	poLayer->ResetReading();

	while ((poFeature = poLayer->GetNextFeature()) != NULL)
	{
		//获取该层的属性信息
		OGRFeatureDefn *poFDefn = poLayer->GetLayerDefn();
		int iField;
		for (iField = 0; iField < poFDefn->GetFieldCount(); iField++) {
			//获取某一个字段的信息
			OGRFieldDefn *poFieldDefn = poFDefn->GetFieldDefn(iField);
			if (poFieldDefn->GetType() == OFTInteger)
				printf("%d,", poFeature->GetFieldAsInteger(iField));
		}
		OGRGeometry *poGeometry;

		//获取feature
		poGeometry = poFeature->GetGeometryRef();
		//用wkbFlatten宏把wkbPoint25D类型转换为wkbPoint类型
		if (poGeometry != NULL
			&& wkbFlatten(poGeometry->getGeometryType()) == wkbPoint) {
			OGRPoint *poPoint = (OGRPoint *)poGeometry;
			printf("%.3f,%3.f\n", poPoint->getX(), poPoint->getY());
		}
		else
			printf("no point geometry\n");
		//销毁feature
		OGRFeature::DestroyFeature(poFeature);
	}
	//销毁数据源，以便关闭矢量文件
	OGRDataSource::DestroyDataSource(poDS);
}
*/


/*


template<typename T>
QuardTree<T>::QuardTree()
{
	root = NULL;
}

template<typename T>
QuardTree<T>::~QuardTree()
{
	clear(root);
}

template<typename T>
int QuardTree<T>::TPLS()
{
	return Depth(root);
}

template<typename T>
int QuardTree<T>::Compare(const Node<T>* node, const Point<T>& pos)
{
	if (pos.x == node->pt.x && pos.z == node->pt.z) return 0;
	if (pos.x >= node->pt.x && pos.z>node->pt.z)  return 1;
	if (pos.x<node->pt.x  && pos.z >= node->pt.z) return 2;
	if (pos.x <= node->pt.x && pos.z<node->pt.z)  return 3;
	if (pos.x>node->pt.x  && pos.z <= node->pt.z) return 4;
	return -1;
}


template<typename T>
void QuardTree<T>::BalanceInsert(const Point<T>& pos)
{
	Node<T>* node = (Node<T>*)malloc(sizeof(Node<T>));
	node->R[0] = NULL;
	node->R[1] = NULL;
	node->R[2] = NULL;
	node->R[3] = NULL;
	node->parent = NULL;
	node->pt = pos;
	if (root == NULL)
	{
		root = node;
		return;
	}
	Node<T>* temp = root;
	int direction = Compare(temp, pos);
	if (direction == 0) return;
	while (temp->R[direction - 1] != NULL)
	{
		temp = temp->R[direction - 1];
		direction = Compare(temp, pos);
		if (direction == 0) return;
	}
	temp->R[direction - 1] = node;
	node->parent = temp;

	Node<T>* tp = temp->parent;
	if (tp == NULL) return;
	int r = Compare(tp, temp->pt);

	if (abs(direction - r) == 2)
	{
		Node<T>* leaf = node;
		if (tp->R[abs(3 - r)] == NULL)
		{
			tp->R[r - 1] = NULL;
			temp->parent = leaf;
			leaf->R[r - 1] = temp;

			temp->R[abs(3 - r)] = NULL;
			Node<T>* Rt = tp->parent;
			if (Rt == NULL)
			{
				root = leaf;
				leaf->parent = NULL;

				leaf->R[abs(3 - r)] = tp;
				tp->parent = leaf;
				return;
			}
			tp->parent = NULL;
			int dd = Compare(Rt, tp->pt);

			Rt->R[dd - 1] = leaf;
			leaf->parent = Rt;

			leaf->R[abs(3 - r)] = tp;
			tp->parent = leaf;
		}
	}
}


template<typename T>
void QuardTree<T>::Insert(Node<T>*& p, const Point<T>& pos)
{
	if (p == NULL)
	{
		Node<T>* node = (Node<T>*)malloc(sizeof(Node<T>));
		node->R[0] = NULL;
		node->R[1] = NULL;
		node->R[2] = NULL;
		node->R[3] = NULL;
		node->pt = pos;
		p = node;
		return;
	}
	else
	{
		int d = Compare(p, pos);
		if (d == 0) return;
		Insert(p->R[d - 1], pos);
	}
}


template<typename T>
void QuardTree<T>::Insert(const Point<T>& pos)
{
	int direction, len = 0;
	Node<T>* node = (Node<T>*)malloc(sizeof(Node<T>));
	node->R[0] = NULL;
	node->R[1] = NULL;
	node->R[2] = NULL;
	node->R[3] = NULL;
	node->pt = pos;
	if (root == NULL)
	{
		root = node;
		return;
	}
	direction = Compare(root, pos);
	Node<T>* temp = root;
	if (direction == 0) return;//节点已存在 
	len = 1;
	while (temp->R[direction - 1] != NULL)
	{
		temp = temp->R[direction - 1];
		direction = Compare(temp, pos);
		if (direction == 0) return;
	}
	temp->R[direction - 1] = node;
	//Insert(root, pos);//递归插入节点
}





template<typename T>
int QuardTree<T>::nodeCount()
{
	return nodeCount(root);
}

template<typename T>
int QuardTree<T>::nodeCount(const Node<T>* node)
{
	if (node == NULL) return 0;
	return 1 + nodeCount(node->R[0]) + nodeCount(node->R[1]) + nodeCount(node->R[2]) + nodeCount(node->R[3]);
}

template<typename T>
bool QuardTree<T>::In_Region(Point<T> t, T left, T right, T botom, T top)
{
	return t.x >= left && t.x <= right && t.y >= botom && t.y <= top;
}

template<typename ElemType>
bool QuardTree<ElemType>::Rectangle_Overlapse_Region(ElemType L, ElemType R, ElemType B, ElemType T,
	ElemType left, ElemType right, ElemType botom, ElemType top)
{
	return L <= right && R >= left && B <= top && T >= botom;
	//return true;
}//优化查找速度

template<typename T>
void QuardTree<T>::RegionResearch(Node<T>* t, T left, T right, T botom, T top, int& visitednum, int& foundnum)
{
	if (t == NULL) return;
	T xc = t->pt.x;
	T zc = t->pt.z;
	if (In_Region(t->pt, left, right, botom, top)) { ++foundnum; }
	if (t->R[0] != NULL && Rectangle_Overlapse_Region(xc, right, zc, top, left, right, botom, top))
	{
		visitednum++;
		RegionResearch(t->R[0], xc>left ? xc : left, right, zc>botom ? zc : botom, top, visitednum, foundnum);
	}
	if (t->R[1] != NULL && Rectangle_Overlapse_Region(left, xc, zc, top, left, right, botom, top))
	{
		visitednum++;
		RegionResearch(t->R[1], left, xc>right ? right : xc, zc>botom ? zc : botom, top, visitednum, foundnum);
	}
	if (t->R[2] != NULL && Rectangle_Overlapse_Region(left, xc, botom, zc, left, right, botom, top))
	{
		visitednum++;
		RegionResearch(t->R[2], left, xc<right ? xc : right, botom, zc<top ? zc : top, visitednum, foundnum);
	}
	if (t->R[3] != NULL && Rectangle_Overlapse_Region(xc, right, botom, zc, left, right, botom, top))
	{
		visitednum++;
		RegionResearch(t->R[3], xc>left ? xc : left, right, botom, zc<top ? zc : top, visitednum, foundnum);
	}
}

template<typename T>
void QuardTree<T>::clear()
{
	clear(root);
}

template<typename T>
void QuardTree<T>::clear(Node<T>* &p)
{
	if (p == NULL) return;
	if (p->R[0]) clear(p->R[0]);
	if (p->R[1]) clear(p->R[1]);
	if (p->R[2]) clear(p->R[2]);
	if (p->R[3]) clear(p->R[3]);
	free(p);
	p = NULL;
}

template<typename T>
void QuardTree<T>::RegionResearch(T left, T right, T botom, T top, int& visitednum, int& foundnum)
{
	RegionResearch(root, left, right, botom, top, visitednum, foundnum);
}

template<typename T>
int QuardTree<T>::Depth(Node<T>* &node)
{
	if (node == NULL) return 0;
	int dep = 0;
	Node<T>* tp = root;
	while (tp->pt.x != node->pt.x || tp->pt.z != node->pt.z)
	{
		dep++;
		tp = tp->R[Compare(tp, node->pt) - 1];
		if (tp == NULL) break;
	}
	return dep + Depth(node->R[0]) + Depth(node->R[1]) + Depth(node->R[2]) + Depth(node->R[3]);
}


int OnChangeByteOrder(int indata)
{
	char ss[9];
	char ee[8];
	unsigned long val = unsigned long(indata);
	ultoa(val, ss, 16);// 将十六进制的数(val)转到一个字符串(ss)中,itoa(val,ss,16); 
	int i;
	int length = strlen(ss);
	if (length != 8) {
		for (i = 0; i<8 - length; i++)
			ee[i] = '0';
		for (i = 0; i<length; i++)
			ee[i + 8 - length] = ss[i];
		for (i = 0; i<8; i++)
			ss[i] = ee[i];
	}
	//****** 进行倒序 
	int t;
	t = ss[0]; ss[0] = ss[6]; ss[6] = t;
	t = ss[1]; ss[1] = ss[7]; ss[7] = t;
	t = ss[2]; ss[2] = ss[4]; ss[4] = t;
	t = ss[3]; ss[3] = ss[5]; ss[5] = t;

	//****** 将存有十六进制数 (val) 的字符串 (ss) 中的十六进制数转成十进制数 
	int value = 0;
	for (i = 0; i<8; i++) {
		int k;
		if (ss[i] == 'a' || ss[i] == 'b' || ss[i] == 'c' || ss[i] == 'd' || ss[i] == 'e' || ss[i] == 'f')
			k = 10 + ss[i] - 'a';
		else
			k = ss[i] - '0';
		value = value + int(k*(int)pow((DOUBLE)16, 7 - i));
	}
	return(value);
}

void readShp(void)
{
	//****打开坐标文件
	CFileDialog fDLG(true);
	if (fDLG.DoModal() != IDOK)
		return;
	filename = fDLG.GetPathName();
	FILE* m_ShpFile_fp = fopen(filename, "rb");
	if (m_ShpFile_fp == NULL) {
		MessageBox("Open File Failed");
		exit(0);
	}

	CGeoMap* map = new CGeoMap();      //创建地图对象
	CGeoLayer* layer = new CGeoLayer();//新建图层                         

									   //****定义读取坐标文件头的变量
	int i;
	int FileCode = -1;
	int Unused = -1;
	int FileLength = -1;
	int Version = -1;
	int ShapeType = -1;
	double Xmin;
	double Ymin;
	double Xmax;
	double Ymax;
	double Zmin;
	double Zmax;
	double Mmin;
	double Mmax;

	//****读取坐标头文件
	fread(&FileCode, sizeof(int), 1, m_ShpFile_fp);  //从m_ShpFile_fp里面的值读到Filecode里面去，每次读一个int型字节的长度，读取一次 
	FileCode = OnChangeByteOrder(FileCode);          //将读取的FileCode的值转化为十进制的数 
	for (i = 0; i<5; i++)
		fread(&Unused, sizeof(int), 1, m_ShpFile_fp);
	fread(&FileLength, sizeof(int), 1, m_ShpFile_fp);//读取FileLength 
	FileLength = OnChangeByteOrder(FileLength);      //将FileLength转化为十进制的数 
	fread(&Version, sizeof(int), 1, m_ShpFile_fp);   //读取Version的值 
	fread(&ShapeType, sizeof(int), 1, m_ShpFile_fp);//读取ShapeType的值 
	fread(&Xmin, sizeof(double), 1, m_ShpFile_fp);//从m_ShpFile_fp里面的值读到Xmin里面去，每次读取一个double型字节长度，读取一次 
	fread(&Ymin, sizeof(double), 1, m_ShpFile_fp);//从m_ShpFile_fp里面的值读到Ymin里面去，每次读取一个double型字节长度，读取一次 
	fread(&Xmax, sizeof(double), 1, m_ShpFile_fp);//从m_ShpFile_fp里面的值读到Xmax里面去，每次读取一个double型字节长度，读取一次 
	fread(&Ymax, sizeof(double), 1, m_ShpFile_fp);//从m_ShpFile_fp里面的值读到Ymax里面去，每次读取一个double型字节长度，读取一次 

	CRect rect(Xmin, Ymin, Xmax, Ymax);
	layer->setRect(rect);                         //设置图层的边界

	fread(&Zmin, sizeof(double), 1, m_ShpFile_fp);//从m_ShpFile_fp里面的值读到Zmin里面去，每次读取一个double型字节长度，读取一次 
	fread(&Zmax, sizeof(double), 1, m_ShpFile_fp);//从m_ShpFile_fp里面的值读到Zmax里面去，每次读取一个double型字节长度，读取一次 
	fread(&Mmin, sizeof(double), 1, m_ShpFile_fp);//从m_ShpFile_fp里面的值读到Mmin里面去，每次读取一个double型字节长度，读取一次 
	fread(&Mmax, sizeof(double), 1, m_ShpFile_fp);//从m_ShpFile_fp里面的值读到Mmax里面去，每次读取一个double型字节长度，读取一次 
												  //****读取坐标文件头的内容结束

												  //****读取面状目标的实体信息
	int RecordNumber;
	int ContentLength;
	switch (ShapeType) {
	case 5: {  //polygon
		while ((fread(&RecordNumber, sizeof(int), 1, m_ShpFile_fp) != 0)) { //从第一个开始循环读取每一个Polygon
			fread(&ContentLength, sizeof(int), 1, m_ShpFile_fp);            //读取ContentLength
			RecordNumber = OnChangeByteOrder(RecordNumber);                 //转换为10进制
			ContentLength = OnChangeByteOrder(ContentLength);
			//****记录头读取结束

			//****读取记录内容
			int shapeType;   //当前Polygon几何类型
			double Box[4];   //当前Polygon的上下左右边界
			int NumParts;    //当前Polygon所包含的子环的个数
			int NumPoints;   //当前Polygon所包含的点的个数
			int *Parts;      //当前Polygon所包含的子环的起点在NumPoints的编号
			fread(&shapeType, sizeof(int), 1, m_ShpFile_fp);
			for (i = 0; i < 4; i++)                         //读Box
				fread(Box + i, sizeof(double), 1, m_ShpFile_fp);
			fread(&NumParts, sizeof(int), 1, m_ShpFile_fp); //表示构成当前Polygon的子环的个数
			fread(&NumPoints, sizeof(int), 1, m_ShpFile_fp);//表示构成当前Polygon所包含的坐标点个数
			Parts = new int[NumParts];                      //记录了每个子环的坐标在Points数组中的起始位置
			for (i = 0; i < NumParts; i++)
				fread(Parts + i, sizeof(int), 1, m_ShpFile_fp);

			int pointNum;
			CGeoPolygon* polygon = new CGeoPolygon();
			polygon->circleNum = NumParts;                   //设定多边形的点数

															 //****读取面中子环
			for (i = 0; i < NumParts; i++) {
				if (i != NumParts - 1)  pointNum = Parts[i + 1] - Parts[i];//每个子环的长度 ，非最后一个环
				else  pointNum = NumPoints - Parts[i];       //最后一个环
				double* PointsX = new double[pointNum];      //用于存放读取的点坐标x值;
				double* PointsY = new double[pointNum];      //用于存放y值
				CGeoPolyline* polyline = new CGeoPolyline(); //每个环实际上就是首尾坐标相同的Polyline
				polyline->circleID = i;

				for (int j = 0; j < pointNum; j++) {
					fread(PointsX + j, sizeof(double), 1, m_ShpFile_fp);
					fread(PointsY + j, sizeof(double), 1, m_ShpFile_fp);
					double a = PointsX[j];
					double b = PointsY[j];
					CPoint* point = new CPoint(a, b);
					polyline->AddPoint(point);               //把该子环所有的点添加到一条链上
				}

				CPoint pt1 = polyline->pts[0]->GetCPoint();
				CPoint pt2 = polyline->pts[polyline->pts.size() - 1]->GetCPoint();
				if (pt1 != pt2) {  //若首位点不一致
					CString str;
					str.Format("%d数据首尾点不一致", RecordNumber);
					polyline->pts.push_back(p1);
				}
				polygon->AddCircle(polyline);                 //将polyline链添加到对应polygon中
				delete[] PointsX;
				delete[] PointsY;
			}
			//****一个面的某个子环循环结束，同时该子环已加入到polygon

			layer->AddObject((CGeoObject*)polygon);           //将该polygon加入到图层中
			delete[] Parts;
		}
		map->AddLayer(layer);
	}
			break;
	case 1://point
		break;
	case 3://polyline
		break;
	default:
		break;
	}
}


FeatureClass* CGISMapDoc::ImportShapeFileData(FILE* fpShp, FILE* fpDbf)
{
	//读Shp文件头开始  
	int fileCode = -1;
	int fileLength = -1;
	int version = -1;
	int shapeType = -1;
	fread(&fileCode, sizeof(int), 1, fpShp);
	fileCode = ReverseBytes(fileCode);

	if (fileCode != 9994)
	{
		CString strTemp;
		strTemp.Format(" WARNING filecode %d ", fileCode);
		AfxMessageBox(strTemp);
	}

	for (int i = 0; i < 5; i++)
		fread(&fileCode, sizeof(int), 1, fpShp);

	fread(&fileLength, sizeof(int), 1, fpShp);
	fileLength = ReverseBytes(fileLength);

	fread(&version, sizeof(int), 1, fpShp);
	fread(&shapeType, sizeof(int), 1, fpShp);

	double tempOriginX, tempOriginY;
	fread(&tempOriginX, sizeof(double), 1, fpShp);
	fread(&tempOriginY, sizeof(double), 1, fpShp);

	double xMaxLayer, yMaxLayer;
	fread(&xMaxLayer, sizeof(double), 1, fpShp);
	fread(&yMaxLayer, sizeof(double), 1, fpShp);

	double* skip = new double[4];
	fread(skip, sizeof(double), 4, fpShp);
	delete[]skip;
	skip = 0;
	//读Shp文件头结束  

	int uniqueID = this->m_pDataSource->GetUniqueID();
	FeatureClass* pShpDataSet = 0;
	//根据目标类型创建相应的图层DataSet。  
	switch (shapeType)
	{
	case 1:
		pShpDataSet = (FeatureClass*)&(m_pDataSource->CreateDataSet(uniqueID, POINTDATASET, layerName));
		break;
	case 3:
	case 23:
		pShpDataSet = (FeatureClass*)&(m_pDataSource->CreateDataSet(uniqueID, LINEDATASET, layerName));
		break;
	case 5:
		pShpDataSet = (FeatureClass*)&(m_pDataSource->CreateDataSet(uniqueID, POLYGONDATASET, layerName));
		break;
	}

	if (pShpDataSet == 0) return 0;

	// 读DBF文件头---------begin------------  
	struct DBFHeader
	{
		char m_nValid;
		char m_aDate[3];
		char m_nNumRecords[4];
		char m_nHeaderBytes[2];
		char m_nRecordBytes[2];
		char m_nReserved1[3];
		char m_nReserved2[13];
		char m_nReserved3[4];
	}dbfheader;

	struct DBFFIELDDescriptor
	{
		char m_sName[10];//应该为char m_sName[11]  
		char m_nType;
		char m_nAddress[4];
		char m_nFieldLength;
		char m_nFieldDecimal;
		char m_nReserved1[2];
		char m_nWorkArea;
		char m_nReserved2[2];
		char m_nSetFieldsFlag;
		char m_nReserved3[8];
	};

	fread(&dbfheader, sizeof(DBFHeader), 1, fpDbf);
	/*int recordsNum = *((int*)dbfheader.m_nNumRecords);
	int headLen = *((short*)dbfheader.m_nHeaderBytes);
	int everyRecordLen = *((short*)dbfheader.m_nRecordBytes);

	if ( recordsNum == 0 ||  headLen == 0 || everyRecordLen == 0 )
	return 0 ;

	int fieldCount = (headLen - 1 - sizeof(DBFHeader))/sizeof(DBFFIELDDescriptor);

	DBFFIELDDescriptor *pFields = new DBFFIELDDescriptor[fieldCount];
	for ( i = 0; i < fieldCount; i ++ )
	fread(&pFields[i],sizeof(DBFFIELDDescriptor),1,fpDbf);

	char endByte;
	fread(&endByte,sizeof(char),1,fpDbf);

	if ( endByte != 0x0D)
	{
	delete []pFields;
	pFields = 0;
	return 0;
	}


	Fields& fields = pShpDataSet->GetFields();
	DBFFIELDDescriptor field;
	BYTE endByte = ' ';
	char fieldName[12];
	int fieldDecimal, fieldLen, everyRecordLen = 0;
	while (!feof(fpDbf))
	{
		fread(&endByte, sizeof(BYTE), 1, fpDbf);
		if (endByte == 0x0D)   break;
		fread(&field, sizeof(DBFFIELDDescriptor), 1, fpDbf);

		fieldName[0] = endByte;
		for (i = 0; i < 10; i++)
			fieldName[i + 1] = field.m_sName[i];
		fieldName[11] = '/0';

		fieldDecimal = field.m_nFieldDecimal;
		fieldLen = field.m_nFieldLength;
		switch (field.m_nType)
		{
		case 'C':
			fields.AddField(fieldName, fieldName, FIELD_STRING, fieldLen);
			break;
		case 'F':
			fields.AddField(fieldName, fieldName, FIELD_DOUBLE, fieldLen);
			break;
		case 'N':
		{
			if (fieldDecimal == 0)
				fields.AddField(fieldName, fieldName, FIELD_INT, fieldLen);
			else fields.AddField(fieldName, fieldName, FIELD_DOUBLE, fieldLen);
		}
		break;
		}
		everyRecordLen += fieldLen;
	}
	// 读DBF文件头---------end------------  

	while (!feof(fpShp))
	{
		//读记录头开始  
		int recordNumber = -1;
		int contentLength = -1;
		fread(&recordNumber, sizeof(int), 1, fpShp);
		fread(&contentLength, sizeof(int), 1, fpShp);
		recordNumber = ReverseBytes(recordNumber);
		contentLength = ReverseBytes(contentLength);
		//读记录头结束  

		switch (shapeType)
		{
		case 1: // '/001'  
				//读取点目标开始  
		{
			Fields &featureFields = pShpDataSet->GetFields();
			Feature *pFeature = new Feature(recordNumber, 1, &featureFields);

			int pointShapeType;
			fread(&pointShapeType, sizeof(int), 1, fpShp);
			double xValue, yValue;
			fread(&xValue, sizeof(double), 1, fpShp);
			fread(&yValue, sizeof(double), 1, fpShp);

			GeoPoint *pNewGeoPoint = new GeoPoint(xValue, yValue);
			pFeature->SetBound(xValue, yValue, 0, 0);
			pFeature->SetGeometry(pNewGeoPoint);
			this->LoadAttributeData(pFeature, fpDbf, everyRecordLen);
			pShpDataSet->AddRow(pFeature);
		}
		break;
		//读取点目标结束  

		case 3: // '/003'  
				//读取线目标开始  
		{
			Fields &featureFields = pShpDataSet->GetFields();
			Feature *pFeature = new Feature(recordNumber, 1, &featureFields);

			int arcShapeType;
			fread(&arcShapeType, sizeof(int), 1, fpShp);

			double objMinX, objMinY, objMaxX, objMaxY;
			fread(&objMinX, sizeof(double), 1, fpShp);
			fread(&objMinY, sizeof(double), 1, fpShp);
			fread(&objMaxX, sizeof(double), 1, fpShp);
			fread(&objMaxY, sizeof(double), 1, fpShp);

			GeoPolyline *pNewGeoLine = new GeoPolyline();
			double width = objMaxX - objMinX;
			double height = objMaxY - objMinY;
			pFeature->SetBound(objMinX, objMinY, width, height);

			int numParts, numPoints;
			fread(&numParts, sizeof(int), 1, fpShp);
			fread(&numPoints, sizeof(int), 1, fpShp);
			//存储各段线的起点索引  
			int* startOfPart = new int[numParts];
			for (int i = 0; i < numParts; i++)
			{
				int indexFirstPoint;
				fread(&indexFirstPoint, sizeof(int), 1, fpShp);
				startOfPart[i] = indexFirstPoint;
			}

			//处理单个目标有多条线的问题  
			pNewGeoLine->SetPointsCount(numParts);

			for (i = 0; i < numParts; i++)
			{
				GeoPoints& points = pNewGeoLine->GetPoints(i);
				int curPosIndex = startOfPart[i];
				int nextPosIndex = 0;
				int curPointCount = 0;
				if (i == numParts - 1)
					curPointCount = numPoints - curPosIndex;
				else
				{
					nextPosIndex = startOfPart[i + 1];
					curPointCount = nextPosIndex - curPosIndex;
				}
				points.SetPointCount(curPointCount);
				//加载一条线段的坐标  
				for (int iteratorPoint = 0; iteratorPoint < curPointCount; iteratorPoint++)
				{
					double x, y;
					fread(&x, sizeof(double), 1, fpShp);
					fread(&y, sizeof(double), 1, fpShp);
					GeoPoint newVertex(x, y);
					points.SetPoint(iteratorPoint, newVertex);
				}
			}
			delete[]startOfPart;
			startOfPart = 0;
			pFeature->SetGeometry(pNewGeoLine);
			this->LoadAttributeData(pFeature, fpDbf, everyRecordLen);
			pShpDataSet->AddRow(pFeature);
		}
		break;
		//读取线目标结束  

		case 5: // '/005'  
				//读取面目标开始  
		{
			Fields &featureFields = pShpDataSet->GetFields();
			Feature *pFeature = new Feature(recordNumber, 1, &featureFields);
			int polygonShapeType;
			fread(&polygonShapeType, sizeof(int), 1, fpShp);
			if (polygonShapeType != 5)
				AfxMessageBox("Error: Attempt to load non polygon shape as polygon.");

			double objMinX, objMinY, objMaxX, objMaxY;
			fread(&objMinX, sizeof(double), 1, fpShp);
			fread(&objMinY, sizeof(double), 1, fpShp);
			fread(&objMaxX, sizeof(double), 1, fpShp);
			fread(&objMaxY, sizeof(double), 1, fpShp);

			GeoPolygon *pNewGeoPolygon = new GeoPolygon();
			double width = objMaxX - objMinX;
			double height = objMaxY - objMinY;
			pFeature->SetBound(objMinX, objMinY, width, height);

			int numParts, numPoints;
			fread(&numParts, sizeof(int), 1, fpShp);
			fread(&numPoints, sizeof(int), 1, fpShp);
			//存储各个面的起点索引  
			int* startOfPart = new int[numParts];
			for (int i = 0; i < numParts; i++)
			{
				int indexFirstPoint;
				fread(&indexFirstPoint, sizeof(int), 1, fpShp);
				startOfPart[i] = indexFirstPoint;
			}

			//处理单个目标有多面问题  
			pNewGeoPolygon->SetPointsCount(numParts);

			for (i = 0; i < numParts; i++)
			{
				GeoPoints& points = pNewGeoPolygon->GetPoints(i);
				int curPosIndex = startOfPart[i];
				int nextPosIndex = 0;
				int curPointCount = 0;
				if (i == numParts - 1)
					curPointCount = numPoints - curPosIndex;
				else
				{
					nextPosIndex = startOfPart[i + 1];
					curPointCount = nextPosIndex - curPosIndex;
				}
				points.SetPointCount(curPointCount);
				//加载一个面(多边形)的坐标  
				for (int iteratorPoint = 0; iteratorPoint < curPointCount; iteratorPoint++)
				{
					double x, y;
					fread(&x, sizeof(double), 1, fpShp);
					fread(&y, sizeof(double), 1, fpShp);
					GeoPoint newVertex(x, y);
					points.SetPoint(iteratorPoint, newVertex);
				}
			}
			delete[]startOfPart;
			startOfPart = 0;
			pFeature->SetGeometry(pNewGeoPolygon);
			this->LoadAttributeData(pFeature, fpDbf, everyRecordLen);
			pShpDataSet->AddRow(pFeature);
		}
		break;
		//读取面目标结束  

		case 23: // '/027'  
				 //读取Measure形线目标开始  
		{
			Fields &featureFields = pShpDataSet->GetFields();
			Feature *pFeature = new Feature(recordNumber, 1, &featureFields);
			int arcMShapeType;
			fread(&arcMShapeType, sizeof(int), 1, fpShp);

			double objMinX, objMinY, objMaxX, objMaxY;
			fread(&objMinX, sizeof(double), 1, fpShp);
			fread(&objMinY, sizeof(double), 1, fpShp);
			fread(&objMaxX, sizeof(double), 1, fpShp);
			fread(&objMaxY, sizeof(double), 1, fpShp);

			GeoPolyline *pNewGeoLine = new GeoPolyline();
			double width = objMaxX - objMinX;
			double height = objMaxY - objMinY;
			pFeature->SetBound(objMinX, objMinY, width, height);

			int numParts, numPoints;
			fread(&numParts, sizeof(int), 1, fpShp);
			fread(&numPoints, sizeof(int), 1, fpShp);
			//存储各段线的起点索引  
			int* startOfPart = new int[numParts];
			for (int i = 0; i < numParts; i++)
			{
				int indexFirstPoint;
				fread(&indexFirstPoint, sizeof(int), 1, fpShp);
				startOfPart[i] = indexFirstPoint;
			}

			//处理单个目标有多条线的问题  
			pNewGeoLine->SetPointsCount(numParts);

			for (i = 0; i < numParts; i++)
			{
				GeoPoints& points = pNewGeoLine->GetPoints(i);
				int curPosIndex = startOfPart[i];
				int nextPosIndex = 0;
				int curPointCount = 0;
				if (i == numParts - 1)
					curPointCount = numPoints - curPosIndex;
				else
				{
					nextPosIndex = startOfPart[i + 1];
					curPointCount = nextPosIndex - curPosIndex;
				}
				points.SetPointCount(curPointCount);
				//加载一条线段的坐标  
				for (int iteratorPoint = 0; iteratorPoint < curPointCount; iteratorPoint++)
				{
					double x, y;
					fread(&x, sizeof(double), 1, fpShp);
					fread(&y, sizeof(double), 1, fpShp);
					GeoPoint newVertex(x, y);
					points.SetPoint(iteratorPoint, newVertex);
				}
			}
			delete[]startOfPart;
			startOfPart = 0;

			double* value = new double[2 + numPoints];
			fread(value, sizeof(double), 2 + numPoints, fpShp);
			delete[]value;
			value = 0;

			pFeature->SetGeometry(pNewGeoLine);
			this->LoadAttributeData(pFeature, fpDbf, everyRecordLen);
			pShpDataSet->AddRow(pFeature);
		}
		break;
		//读取Measure形线目标结束  
		}
	}
	return pShpDataSet;
}*/