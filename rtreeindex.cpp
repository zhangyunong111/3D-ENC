#include "rtreeindex.h"

RTreeIndex::RTreeIndex() { this->root = new RTreeLeafNode(); }

RTreeNode *RTreeIndex::Insert(Features<float> *Feature, OGREnvelope &nodeEnvelope, int &RT_k) {
	this->root = this->root->Insert(Feature, nodeEnvelope, RT_k);
	return this->root;
}

// �������нڵ��MBR�����ɽڵ�����
vector<float> RTreeIndex::Skeleton() {
	vector<OGREnvelope> skeleton;
	this->root->Skeleton(skeleton);
	vector<float> vertex;
	for (OGREnvelope &envelope : skeleton) {
		vertex.push_back (envelope.MinX);
		vertex.push_back (envelope.MinY);

		vertex.push_back (envelope.MinX);
		vertex.push_back (envelope.MaxY);

		vertex.push_back (envelope.MaxX);
		vertex.push_back (envelope.MaxY);

		vertex.push_back (envelope.MaxX);
		vertex.push_back (envelope.MinY);
	}
	return vertex;
}

int RTreeBranchNode::Count() { return this->childs.size(); }

int RTreeLeafNode::Count() { return this->values.size(); }

// �����ӽڵ��Ҫ�ش洢��˳�򣬷������ǵ�MBR
vector<OGREnvelope> RTreeBranchNode::SubEnvelope() {
	vector<OGREnvelope> subs;
	//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", this->envelope.MaxX, this->envelope.MaxY, this->envelope.MinX, this->envelope.MinY);

	for (RTreeNode *child : childs) {
		//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", child->envelope.MaxX, child->envelope.MaxY, child->envelope.MinX, child->envelope.MinY);

		subs.push_back(child->envelope);
	}
	return subs;
}

vector<OGREnvelope> RTreeLeafNode::SubEnvelope() {
	vector<OGREnvelope> subs;		
	for (Features<float> *value : values) {
		//OGREnvelope mergeEnvelope = OGRUtils::FeatureEnvelopeIntersection(this->envelope, value->ENCenvelope);
		//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", value->ENCenvelope.MaxX, value->ENCenvelope.MaxY, value->ENCenvelope.MinX, value->ENCenvelope.MinY);
		//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", mergeEnvelope.MaxX, mergeEnvelope.MaxY, mergeEnvelope.MinX, mergeEnvelope.MinY);
		//subs.push_back (mergeEnvelope);
		subs.push_back(OGRUtils::FeatureEnvelope(value->ENCFeature));
	}
	return subs;
}

void RTreeBranchNode::Search(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node)
{
	// �ж����������Ƿ���ڵ������غ�
	if (!envelope.Intersects(searchArea)) {
		return;
	}
	node = vector<RTreeNode *>();
	// ����غϣ������ӽڵ�����
	for (RTreeNode *child : childs) {
		child->Search(searchArea, result, node);
	}
	return;
}

void RTreeLeafNode::Search(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node)
{
	// �ж����������Ƿ���ڵ������غ�
	if (!envelope.Intersects(searchArea)) {
		return;
	}

	// ����غϣ�����Ҫ��
	OGREnvelope envelope;
	for (Features<float> *value : values) {
		value->ENCFeature->GetGeometryRef()->getEnvelope(&envelope);
		// �ж����������Ƿ���Ҫ���غ�
		if (envelope.Intersects(searchArea)) {
			result.push_back (value);
		}
	}
	node.push_back (this);
	return;
}

//QRTree��ѯ
void RTreeBranchNode::QRTreeSearch(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node)
{
	for (RTreeNode *child : childs) {
		child->QRTreeSearch(searchArea, result, node);
		//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", child->envelope.MaxX, child->envelope.MaxY, child->envelope.MinX, child->envelope.MinY);

	}
	return;
}
void RTreeLeafNode::QRTreeSearch(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node)
{
		////node.push_back(this);
	if (OGRUtils::Envelope_In_Region(this->envelope, searchArea) == 1)
	{//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", this->envelope.MaxX, this->envelope.MaxY, this->envelope.MinX, this->envelope.MinY);

		node.push_back(this);
		result.insert(result.end(), this->values.begin(), this->values.end());
	}
	else if (OGRUtils::Envelope_In_Region(this->envelope, searchArea) == 2)
	{
		for (Features<float> *value : values) {
			if (value->ENCenvelope.Intersects(searchArea)) {
				result.push_back(value);
			}
		}
		node.push_back(this);
	}
	return;
}

void RTreeBranchNode::GetNode(vector<RTreeNode *> node)
{
	for (RTreeNode *child : childs) {
		child->GetNode(node);
	}
	return ;
}
void RTreeLeafNode::GetNode(vector<RTreeNode *> node)
{
	node.push_back(this);
	return;
}

/*
// ��ȡҪ�ز����Ҷ��㡣����������Сԭ��,SMBB
RTreeLeafNode *RTreeBranchNode::ChooseLeafsmallest(Features<float> *Feature)
{
	// �ҳ����E.Iʱ��������С���ӽڵ�
	// �����뵽�ĸ��ӽڵ��У��ӽڵ����С��Ӿ���������С
	double minDiff = DBL_MAX;
	RTreeNode *minChild;
	OGREnvelope valueEnvelope;
	//value->GetGeometryRef()->getEnvelope(&valueEnvelope);
	for (RTreeNode *child : childs) // ���ｫ������С���Ϊ���������С
	{
		double mergeDiffer = OGRUtils::EnvelopeMergeDiffer(child->envelope, valueEnvelope);
		if (mergeDiffer < minDiff) {
			minDiff = mergeDiffer;
			minChild = child;
		}
	}
	// �ҵ�������С���ӽڵ㣬�ݹ���á��ݹ鵽LeafNodeΪֹ
	return minChild->ChooseLeafsmallest(Feature);
}
*/
// ��ȡҪ�ز����Ҷ��㡣����������Сԭ��
RTreeLeafNode *RTreeBranchNode::ChooseLeaf(OGREnvelope &valueEnvelope)
{
	// �ҳ����E.Iʱ��������С���ӽڵ�
	// �����뵽�ĸ��ӽڵ��У��ӽڵ����С��Ӿ���������С
	double minDiff = DBL_MAX;
	RTreeNode *minChild;
	//OGREnvelope valueEnvelope;
	//value->GetGeometryRef()->getEnvelope(&valueEnvelope);
	for (RTreeNode *child : childs) // ���ｫ������С���Ϊ���������С
	{
		double mergeDiffer = OGRUtils::EnvelopeMergeDiffer(child->envelope, valueEnvelope);
		if (mergeDiffer < minDiff) {
			minDiff = mergeDiffer;
			minChild = child;
		}
	}
	// �ҵ�������С���ӽڵ㣬�ݹ���á��ݹ鵽LeafNodeΪֹ
	return minChild->ChooseLeaf(valueEnvelope);
}

RTreeLeafNode *RTreeLeafNode::ChooseLeaf(OGREnvelope &valueEnvelope) { return this; }

// ʹ��Quadratic�����η�����������ѡ���ѵ��������ӽڵ㡣�������ص�ԭ��
vector<int> RTreeNode::pickSeed() 
{
	vector<int> seed(2);
	vector<OGREnvelope> &&subs = this->SubEnvelope();
	double maxDiff = -1;
	// ÿ��������һ�飬Ѱ�����ź�������������������һ��
	for (int i = 0; i < subs.size(); i++) {
		for (int j = i + 1; j < subs.size(); j++) {
			double mergeDiffer = OGRUtils::EnvelopeMergeDiffer(subs[i], subs[j]);
			if (mergeDiffer > maxDiff) {
				maxDiff = mergeDiffer;
				// ʹ��[]���������������
				seed[0] = i;
				seed[1] = j;
			}
		}
	}
	return seed;
}

// ������ѡ�����ӽڵ㣬���ط���
vector<vector<int>> RTreeNode::seedGroup(vector<int> seed) 
{
	vector<vector<int>> group(2);
	// �����ӽڵ���������
	group[0].push_back (seed[0]);
	group[1].push_back (seed[1]);
	vector<OGREnvelope> &&subs = this->SubEnvelope();
	for (int i = 0; i < subs.size(); i++) {
		// ��������ӽڵ㣬�������ж�
		if (i == seed[0] || i == seed[1]) {
			continue;
		}
		// ����������ӽڵ㣬���������������ӵ�С��һ��
		// ��seed1���������
		double mergeDiffer1 = OGRUtils::EnvelopeMergeDiffer(subs[i], subs[seed[0]]);
		// ��seed2���������
		double mergeDiffer2 = OGRUtils::EnvelopeMergeDiffer(subs[i], subs[seed[1]]);

		if (mergeDiffer1 < mergeDiffer2) {
			group[0].push_back (i);
		}
		else {
			group[1].push_back (i);
		}
	}
	return group;
}

// ���ѽڵ㣬���ط��ѵ��½ڵ�
RTreeNode *RTreeBranchNode::SplitNode() {
	// �����½ڵ㣬�����ݷ��飬����ԭʼ�ڵ���½ڵ���ӽڵ�

	vector<vector<int>> group = this->seedGroup(this->pickSeed());
	// ԭ�ڵ�������ӽڵ�
	vector<RTreeNode *> childs = this->childs;
	this->childs.clear();
	RTreeBranchNode *newNode = new RTreeBranchNode();
	// �뱻���ѽڵ������ͬ�ĸ��ڵ�
	newNode->parent = this->parent;
	//newNode->envelope = this->envelope;//�̳�ԭ�ڵ�ķ�Χ
	for (int i = 0; i < group[0].size(); i++) {
		this->childs.push_back (childs[group[0][i]]);
	}
	for (int i = 0; i < group[1].size(); i++) {
		newNode->childs.push_back (childs[group[1][i]]);
		// �½ڵ���Ҫ�ı��ӽڵ�ĸ��ڵ�ָ��
		childs[group[1][i]]->parent = newNode;
	}
	return newNode;
}

RTreeNode *RTreeLeafNode::SplitNode() {
	// �����½ڵ㣬�����ݷ��飬����ԭʼ�ڵ���½ڵ��ֵ

	vector<vector<int>> group = this->seedGroup(this->pickSeed());
	// ԭ�ڵ������ֵ
	vector<Features<float> *> values = this->values;
	this->values.clear();
	RTreeLeafNode *newNode = new RTreeLeafNode();
	newNode->parent = this->parent;
	//newNode->envelope = this->envelope;//�̳�ԭ�ڵ�ķ�Χ
	for (int i = 0; i < group[0].size(); i++) {
		this->values.push_back (values[group[0][i]]);
	}
	for (int i = 0; i < group[1].size(); i++) {
		newNode->values.push_back (values[group[1][i]]);
	}
	this->AdjustEnvelope();
	newNode->AdjustEnvelope();
	return newNode;
}

void RTreeNode::AdjustEnvelope() {
	OGREnvelope currentEnvelope;
	
	vector<OGREnvelope> &&subs = this->SubEnvelope();
	for (OGREnvelope sub : subs) {
		currentEnvelope.Merge(sub);
	}
	//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", this->envelope.MaxX, this->envelope.MaxY, this->envelope.MinX, this->envelope.MinY);
	//currentEnvelope = OGRUtils::FeatureEnvelopeIntersection(this->envelope, currentEnvelope);
	this->envelope = currentEnvelope;
}

// �����������ظ��ڵ㡣�޲�ֻ��MBR����
RTreeNode *RTreeNode::AdjustTree() {
	this->AdjustEnvelope();
	if (parent == nullptr) {
		return this;
	}
	return parent->AdjustTree();
}

// �в�������µķ��ѽڵ������ڵ�
RTreeNode *RTreeNode::AdjustTree(RTreeNode *newNode, int &RT_k) {
	// ���ڵ���ѣ������µĸ��ڵ�
	if (parent == nullptr) {
		RTreeBranchNode *newRoot = new RTreeBranchNode();
		this->parent = newRoot;
		newNode->parent = newRoot;
		newRoot->childs.push_back (this);
		newRoot->childs.push_back (newNode);
		//newRoot->envelope = this->envelope;//�̳�ԭ�ڵ�ķ�Χ
		newRoot->AdjustEnvelope();
		return newRoot;
	}
	// ��ӷ��ѽڵ������ڵ�
	parent->childs.push_back (newNode);
	// TODO:��������
	//���ڵ�ĺ���������RT_kʱҲ��Ҫ����
	if (parent->Count() <= RT_k) {
		parent->AdjustEnvelope();
		return parent->AdjustTree();
	}
	else {
		// ���ڵ���Ѻ󣬿��ܻ�ı䵱ǰ�ڵ���ָ���ڵ�Ϊ�·��ѽڵ�
		// ��˼�¼����ǰ�ĸ��ڵ�
		RTreeNode *oldParent = this->parent;
		RTreeNode *newParentNode = parent->SplitNode();
		// ���ڵ㷢�����ѣ�Ҳ��Ҫ��Ӹ��ڵ�ķ��ѽڵ����丸�ڵ�
		return oldParent->AdjustTree(newParentNode, RT_k);
	}
}

// ���ø�������������Ҫ��
RTreeNode *RTreeNode::Insert(Features<float> *value, OGREnvelope &nodeEnvelope, int &RT_k) {
	//OGRFeature *value;
	//value = Feature->ENCFeature;
	OGREnvelope mergeEnvelope = OGRUtils::FeatureEnvelopeIntersection(nodeEnvelope, value->ENCenvelope);
	//this->envelope = nodeEnvelope;
	//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", this->envelope.MaxX, this->envelope.MaxY, this->envelope.MinX, this->envelope.MinY);

	RTreeLeafNode *leafNode = ChooseLeaf(mergeEnvelope);
	////RTreeLeafNode *leafNode = ChooseLeaf(value->ENCenvelope);
	leafNode->values.push_back (value);
	//leafNode->envelope = mergeEnvelope;
	// TODO:��������
	if (leafNode->values.size() > RT_k) {
		RTreeLeafNode *newLeafNode = (RTreeLeafNode *)leafNode->SplitNode();
		return leafNode->AdjustTree(newLeafNode, RT_k);
	}
	else {
		return leafNode->AdjustTree();
	}
}

// ��ȡ���нڵ��MBR
void RTreeBranchNode::Skeleton(vector<OGREnvelope> &skeleton) {
	skeleton.push_back (this->envelope);
	for (RTreeNode *child : childs) {
		child->Skeleton(skeleton);
	}
}

void RTreeLeafNode::Skeleton(vector<OGREnvelope> &skeleton) {
	skeleton.push_back (this->envelope);
	//skeleton.push_back (this->SubEnvelope());
}


MatrixUtils::MatrixUtils() {}

double OGRUtils::EnvelopeArea(const OGREnvelope &envelope) {
	return (envelope.MaxX - envelope.MinX) * (envelope.MaxY - envelope.MinY);
}

OGREnvelope OGRUtils::FeatureEnvelope(OGRFeature &feature) 
{
	OGREnvelope envelope;
	feature.GetGeometryRef()->getEnvelope(&envelope);
	return envelope;
}

OGREnvelope OGRUtils::FeatureEnvelope(OGRFeature *const feature) 
{
	OGREnvelope envelope;
	feature->GetGeometryRef()->getEnvelope(&envelope);
	return envelope;
}

double OGRUtils::EnvelopeMergeDiffer(const OGREnvelope &envelope1, const OGREnvelope &envelope2)
{
	OGREnvelope mergeEnvelope;
	mergeEnvelope.Merge(envelope1);
	mergeEnvelope.Merge(envelope2);
	return EnvelopeArea(mergeEnvelope) - EnvelopeArea(envelope1) - EnvelopeArea(envelope2);
}

int OGRUtils::Envelope_In_Region(const OGREnvelope &FeatureEnvelope, const OGREnvelope &SearchEnvelope)
{
	if (FeatureEnvelope.MaxX<SearchEnvelope.MaxX && FeatureEnvelope.MinX>SearchEnvelope.MinX && FeatureEnvelope.MaxY<SearchEnvelope.MaxY && FeatureEnvelope.MinY>SearchEnvelope.MinY)
	{
		return 1;//����
	}
	double minx = MAX(FeatureEnvelope.MinX, SearchEnvelope.MinX);
	double miny = MAX(FeatureEnvelope.MinY, SearchEnvelope.MinY);
	double maxx = MIN(FeatureEnvelope.MaxX, SearchEnvelope.MaxX);
	double maxy = MIN(FeatureEnvelope.MaxY, SearchEnvelope.MaxY);
	if (minx > maxx || miny > maxy)
	{
		return 0;//���뽻
	}
	else
	{
		return 2;//�ཻ
	}
}

OGREnvelope OGRUtils::FeatureEnvelopeIntersection(const OGREnvelope &envelope1, const OGREnvelope &envelope2)
{
	OGREnvelope mergeEnvelope = envelope2;
	if (envelope1.MinX == INFINITE)
	{
		return mergeEnvelope;
	}
	else
	{
		mergeEnvelope.Intersect(envelope1);
		return mergeEnvelope;
	}
	
}