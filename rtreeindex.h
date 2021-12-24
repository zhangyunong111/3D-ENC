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
�ڵ㣺
��֧�ڵ㣺ָ���֧�ڵ����Ҷ�ڵ�
Ҷ�ڵ㣺ָ��Ҫ��
*/
class RTreeNode {
public:
	// ���ڵ�
	RTreeBranchNode *parent = NULL;

	virtual int Count() = 0;

	//    vector<RTreeNode*> childs;
	//    vector<OGRFeature*> values;

	// ÿ���ڵ㶼��MBR����ʾ�ڵ�������Ӿ���
	OGREnvelope envelope;

	// �����ӽڵ��Ҫ�ش洢��˳�򣬷������ǵ�MBR
	virtual vector<OGREnvelope> SubEnvelope() = 0;

	// ʹ���麯��ʱ��Ҫע���麯�����������ʵ����
	virtual void Search(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node) = 0;

	// ��ȡҪ�ز����Ҷ��㡣����������Сԭ��
	//virtual RTreeLeafNode *ChooseLeaf(OGRFeature *value) = 0;
	virtual RTreeLeafNode *ChooseLeaf(OGREnvelope &valueEnvelope) = 0;

	// ʹ��Quadratic�����η�����������ѡ���ѵ��������ӽڵ㡣�������ص�ԭ��
	vector<int> pickSeed();

	// ������ѡ�����ӽڵ㣬���ط���
	vector<vector<int>> seedGroup(vector<int> seed);

	// ���ѽڵ㣬���ط��ѵ��½ڵ�
	virtual RTreeNode *SplitNode() = 0;

	void AdjustEnvelope();

	// �����������ظ��ڵ㡣�޲�ֻ��MBR����
	RTreeNode *AdjustTree();

	// �в�������µķ��ѽڵ������ڵ�
	RTreeNode *AdjustTree(RTreeNode *newNode, int &RT_k);

	// ���ø�������������Ҫ��
	RTreeNode *Insert(Features<float> *value, OGREnvelope &nodeEnvelope, int &RT_k);

	// ��ȡ���нڵ��MBR
	virtual void Skeleton(vector<OGREnvelope> &skeleton) = 0;

	//QRTree��ѯ
	virtual void QRTreeSearch(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node) = 0;
	// ��ȡҪ�ز����Ҷ��㡣����������Сԭ��,SMBB
	//virtual RTreeLeafNode *ChooseLeafsmallest(Features<float> *Feature) = 0;
	virtual void GetNode(vector<RTreeNode *> node) = 0;
};

class RTreeBranchNode : public RTreeNode {
public:
	// ��֧�ڵ�����ӽڵ㣬�ӽڵ�����Ƿ�֧�ڵ��Ҷ�ڵ�
	vector<RTreeNode *> childs;

	int Count();

	//OGREnvelope nodeEnvelope;//�ڵ�MBB

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
	// Ҷ�ڵ�û���ӽڵ㣬����������Ҫ��
	vector<Features<float> *> values;

	int Count();

	//OGREnvelope nodeEnvelope;//�ڵ�MBB

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

	// ���ڵ㡣��ʼ��ʱΪҶ��㣬���Ѻ��Ϊ��֧�ڵ�
	RTreeNode *root = nullptr;

	RTreeNode *Insert(Features<float> *Feature, OGREnvelope &nodeEnvelope, int &RT_k);

	// �������нڵ��MBR�����ɽڵ�����
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

	// ��������MBR�ϲ�֮�����������������������
	static double EnvelopeMergeDiffer(const OGREnvelope &envelope1,	const OGREnvelope &envelope2);

	static int Envelope_In_Region(const OGREnvelope &FeatureEnvelope, const OGREnvelope &SearchEnvelope);
	// ��������envelope�ϲ�֮��ķ�Χ
	static OGREnvelope FeatureEnvelopeIntersection(const OGREnvelope &envelope1, const OGREnvelope &envelope2);
};

#endif  // RTREEINDEX_H
