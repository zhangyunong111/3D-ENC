#pragma once

#ifndef RTREEINDEX_H
#define RTREEINDEX_H

#include <vector>
#include <ReadData.h>
using namespace std;

//#include "ogrsf_frmts.h"
//#include "utils.h"

class RTreeBranchNode;

class RTreeLeafNode;

/*
节点：
分支节点：指向分支节点或者叶节点
叶节点：指向要素
*/
class RTreeNode {
public:
	// 父节点
	RTreeBranchNode *parent = NULL;

	virtual int Count() = 0;

	//    vector<RTreeNode*> childs;
	//    vector<OGRFeature*> values;

	// 每个节点都有MBR，表示节点区域、外接矩形
	OGREnvelope envelope;

	// 按照子节点或要素存储的顺序，返回它们的MBR
	virtual vector<OGREnvelope> SubEnvelope() = 0;

	// 使用虚函数时，要注意虚函数的虚调用与实调用
	virtual void Search(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node) = 0;

	// 获取要素插入的叶结点。依据扩张最小原则
	//virtual RTreeLeafNode *ChooseLeaf(OGRFeature *value) = 0;
	virtual RTreeLeafNode *ChooseLeaf(OGREnvelope &valueEnvelope) = 0;

	// 使用Quadratic（二次方）方案，挑选分裂的两个种子节点。尽量不重叠原则
	vector<int> pickSeed();

	// 根据挑选的种子节点，返回分组
	vector<vector<int>> seedGroup(vector<int> seed);

	// 分裂节点，返回分裂的新节点
	virtual RTreeNode *SplitNode() = 0;

	void AdjustEnvelope();

	// 调整树，返回根节点。无参只做MBR调整
	RTreeNode *AdjustTree();

	// 有参则添加新的分裂节点至父节点
	RTreeNode *AdjustTree(RTreeNode *newNode, int &RT_k);

	// 利用辅助方法，插入要素
	RTreeNode *Insert(Features<float> *value, OGREnvelope &nodeEnvelope, int &RT_k);

	// 获取所有节点的MBR
	virtual void Skeleton(vector<OGREnvelope> &skeleton) = 0;

	//QRTree查询
	virtual void QRTreeSearch(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node) = 0;
	// 获取要素插入的叶结点。依据扩张最小原则,SMBB
	//virtual RTreeLeafNode *ChooseLeafsmallest(Features<float> *Feature) = 0;
	virtual void GetNode(vector<RTreeNode *> node) = 0;
};

class RTreeBranchNode : public RTreeNode {
public:
	// 分支节点具有子节点，子节点可能是分支节点或叶节点
	vector<RTreeNode *> childs;

	int Count();

	//OGREnvelope nodeEnvelope;//节点MBB

	vector<OGREnvelope> SubEnvelope() override;

	void Search(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node) override;

	//RTreeLeafNode *ChooseLeaf(OGRFeature *value) override;
	RTreeLeafNode *ChooseLeaf(OGREnvelope &valueEnvelope) override;

	RTreeNode *SplitNode() override;

	void Skeleton(vector<OGREnvelope> &skeleton) override;

	void QRTreeSearch(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node) override;
	//RTreeLeafNode *ChooseLeafsmallest(Features<float> *Feature) override;
	void GetNode(vector<RTreeNode *> node) override;

};

class RTreeLeafNode : public RTreeNode {
public:
	// 叶节点没有子节点，但是它包含要素
	vector<Features<float> *> values;

	int Count();

	//OGREnvelope nodeEnvelope;//节点MBB

	vector<OGREnvelope> SubEnvelope() override;

	void Search(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node) override;

	//RTreeLeafNode *ChooseLeaf(OGRFeature *value) override;
	RTreeLeafNode *ChooseLeaf(OGREnvelope &valueEnvelope) override;

	RTreeNode *SplitNode() override;

	void Skeleton(vector<OGREnvelope> &skeleton) override;

	void QRTreeSearch(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node) override;
	//RTreeLeafNode *ChooseLeafsmallest(Features<float> *Feature) override;
	void GetNode(vector<RTreeNode *> node) override;
};

class RTreeIndex {
public:
	RTreeIndex();

	// 根节点。初始化时为叶结点，分裂后变为分支节点
	RTreeNode *root = nullptr;

	RTreeNode *Insert(Features<float> *Feature, OGREnvelope &nodeEnvelope, int &RT_k);

	// 根据所有节点的MBR，生成节点数组
	vector<float> Skeleton();
};


class MatrixUtils {
public:
	MatrixUtils();
};

class OGRUtils {
public:
	static double EnvelopeArea(const OGREnvelope &envelope);

	static OGREnvelope FeatureEnvelope(OGRFeature &feature);

	static OGREnvelope FeatureEnvelope(OGRFeature *const feature);

	// 计算两个MBR合并之后的面积增量（面积扩张量）
	static double EnvelopeMergeDiffer(const OGREnvelope &envelope1,	const OGREnvelope &envelope2);

	static int Envelope_In_Region(const OGREnvelope &FeatureEnvelope, const OGREnvelope &SearchEnvelope);
	// 计算两个envelope合并之后的范围
	static OGREnvelope FeatureEnvelopeIntersection(const OGREnvelope &envelope1, const OGREnvelope &envelope2);
};

#endif  // RTREEINDEX_H
