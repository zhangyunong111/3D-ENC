#include "rtreeindex.h"

RTreeIndex::RTreeIndex() { this->root = new RTreeLeafNode(); }

RTreeNode *RTreeIndex::Insert(Features<float> *Feature, OGREnvelope &nodeEnvelope, int &RT_k) {
	this->root = this->root->Insert(Feature, nodeEnvelope, RT_k);
	return this->root;
}

// 根据所有节点的MBR，生成节点数组
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

// 按照子节点或要素存储的顺序，返回它们的MBR
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
	// 判断搜索区域是否与节点区域重合
	if (!envelope.Intersects(searchArea)) {
		return;
	}
	node = vector<RTreeNode *>();
	// 如果重合，调用子节点搜索
	for (RTreeNode *child : childs) {
		child->Search(searchArea, result, node);
	}
	return;
}

void RTreeLeafNode::Search(const OGREnvelope &searchArea, vector<Features<float> *> &result, vector<RTreeNode *> &node)
{
	// 判断搜索区域是否与节点区域重合
	if (!envelope.Intersects(searchArea)) {
		return;
	}

	// 如果重合，遍历要素
	OGREnvelope envelope;
	for (Features<float> *value : values) {
		value->ENCFeature->GetGeometryRef()->getEnvelope(&envelope);
		// 判断搜索区域是否与要素重合
		if (envelope.Intersects(searchArea)) {
			result.push_back (value);
		}
	}
	node.push_back (this);
	return;
}

//QRTree查询
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
// 获取要素插入的叶结点。依据扩张最小原则,SMBB
RTreeLeafNode *RTreeBranchNode::ChooseLeafsmallest(Features<float> *Feature)
{
	// 找出添加E.I时，扩张最小的子节点
	// 即加入到哪个子节点中，子节点的最小外接矩形扩张最小
	double minDiff = DBL_MAX;
	RTreeNode *minChild;
	OGREnvelope valueEnvelope;
	//value->GetGeometryRef()->getEnvelope(&valueEnvelope);
	for (RTreeNode *child : childs) // 这里将扩张最小理解为面积增加最小
	{
		double mergeDiffer = OGRUtils::EnvelopeMergeDiffer(child->envelope, valueEnvelope);
		if (mergeDiffer < minDiff) {
			minDiff = mergeDiffer;
			minChild = child;
		}
	}
	// 找到扩张最小的子节点，递归调用。递归到LeafNode为止
	return minChild->ChooseLeafsmallest(Feature);
}
*/
// 获取要素插入的叶结点。依据扩张最小原则
RTreeLeafNode *RTreeBranchNode::ChooseLeaf(OGREnvelope &valueEnvelope)
{
	// 找出添加E.I时，扩张最小的子节点
	// 即加入到哪个子节点中，子节点的最小外接矩形扩张最小
	double minDiff = DBL_MAX;
	RTreeNode *minChild;
	//OGREnvelope valueEnvelope;
	//value->GetGeometryRef()->getEnvelope(&valueEnvelope);
	for (RTreeNode *child : childs) // 这里将扩张最小理解为面积增加最小
	{
		double mergeDiffer = OGRUtils::EnvelopeMergeDiffer(child->envelope, valueEnvelope);
		if (mergeDiffer < minDiff) {
			minDiff = mergeDiffer;
			minChild = child;
		}
	}
	// 找到扩张最小的子节点，递归调用。递归到LeafNode为止
	return minChild->ChooseLeaf(valueEnvelope);
}

RTreeLeafNode *RTreeLeafNode::ChooseLeaf(OGREnvelope &valueEnvelope) { return this; }

// 使用Quadratic（二次方）方案，挑选分裂的两个种子节点。尽量不重叠原则
vector<int> RTreeNode::pickSeed() 
{
	vector<int> seed(2);
	vector<OGREnvelope> &&subs = this->SubEnvelope();
	double maxDiff = -1;
	// 每两子区域一组，寻找扩张后面积与各自面积相差最大的一组
	for (int i = 0; i < subs.size(); i++) {
		for (int j = i + 1; j < subs.size(); j++) {
			double mergeDiffer = OGRUtils::EnvelopeMergeDiffer(subs[i], subs[j]);
			if (mergeDiffer > maxDiff) {
				maxDiff = mergeDiffer;
				// 使用[]运算符，返回引用
				seed[0] = i;
				seed[1] = j;
			}
		}
	}
	return seed;
}

// 根据挑选的种子节点，返回分组
vector<vector<int>> RTreeNode::seedGroup(vector<int> seed) 
{
	vector<vector<int>> group(2);
	// 将种子节点添加入分组
	group[0].push_back (seed[0]);
	group[1].push_back (seed[1]);
	vector<OGREnvelope> &&subs = this->SubEnvelope();
	for (int i = 0; i < subs.size(); i++) {
		// 如果是种子节点，不进行判断
		if (i == seed[0] || i == seed[1]) {
			continue;
		}
		// 如果不是种子节点，计算面积增量，添加到小的一组
		// 与seed1的面积增量
		double mergeDiffer1 = OGRUtils::EnvelopeMergeDiffer(subs[i], subs[seed[0]]);
		// 与seed2的面积增量
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

// 分裂节点，返回分裂的新节点
RTreeNode *RTreeBranchNode::SplitNode() {
	// 创建新节点，并根据分组，设置原始节点和新节点的子节点

	vector<vector<int>> group = this->seedGroup(this->pickSeed());
	// 原节点的所有子节点
	vector<RTreeNode *> childs = this->childs;
	this->childs.clear();
	RTreeBranchNode *newNode = new RTreeBranchNode();
	// 与被分裂节点具有相同的父节点
	newNode->parent = this->parent;
	//newNode->envelope = this->envelope;//继承原节点的范围
	for (int i = 0; i < group[0].size(); i++) {
		this->childs.push_back (childs[group[0][i]]);
	}
	for (int i = 0; i < group[1].size(); i++) {
		newNode->childs.push_back (childs[group[1][i]]);
		// 新节点需要改变子节点的父节点指针
		childs[group[1][i]]->parent = newNode;
	}
	return newNode;
}

RTreeNode *RTreeLeafNode::SplitNode() {
	// 创建新节点，并根据分组，设置原始节点和新节点的值

	vector<vector<int>> group = this->seedGroup(this->pickSeed());
	// 原节点的所有值
	vector<Features<float> *> values = this->values;
	this->values.clear();
	RTreeLeafNode *newNode = new RTreeLeafNode();
	newNode->parent = this->parent;
	//newNode->envelope = this->envelope;//继承原节点的范围
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

// 调整树，返回根节点。无参只做MBR调整
RTreeNode *RTreeNode::AdjustTree() {
	this->AdjustEnvelope();
	if (parent == nullptr) {
		return this;
	}
	return parent->AdjustTree();
}

// 有参则添加新的分裂节点至父节点
RTreeNode *RTreeNode::AdjustTree(RTreeNode *newNode, int &RT_k) {
	// 根节点分裂，生成新的根节点
	if (parent == nullptr) {
		RTreeBranchNode *newRoot = new RTreeBranchNode();
		this->parent = newRoot;
		newNode->parent = newRoot;
		newRoot->childs.push_back (this);
		newRoot->childs.push_back (newNode);
		//newRoot->envelope = this->envelope;//继承原节点的范围
		newRoot->AdjustEnvelope();
		return newRoot;
	}
	// 添加分裂节点至父节点
	parent->childs.push_back (newNode);
	// TODO:更换常量
	//父节点的孩子数大于RT_k时也需要分裂
	if (parent->Count() <= RT_k) {
		parent->AdjustEnvelope();
		return parent->AdjustTree();
	}
	else {
		// 父节点分裂后，可能会改变当前节点所指父节点为新分裂节点
		// 因此记录分裂前的父节点
		RTreeNode *oldParent = this->parent;
		RTreeNode *newParentNode = parent->SplitNode();
		// 父节点发生分裂，也需要添加父节点的分裂节点至其父节点
		return oldParent->AdjustTree(newParentNode, RT_k);
	}
}

// 利用辅助方法，插入要素
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
	// TODO:更换常量
	if (leafNode->values.size() > RT_k) {
		RTreeLeafNode *newLeafNode = (RTreeLeafNode *)leafNode->SplitNode();
		return leafNode->AdjustTree(newLeafNode, RT_k);
	}
	else {
		return leafNode->AdjustTree();
	}
}

// 获取所有节点的MBR
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
		return 1;//包含
	}
	double minx = MAX(FeatureEnvelope.MinX, SearchEnvelope.MinX);
	double miny = MAX(FeatureEnvelope.MinY, SearchEnvelope.MinY);
	double maxx = MIN(FeatureEnvelope.MaxX, SearchEnvelope.MaxX);
	double maxy = MIN(FeatureEnvelope.MaxY, SearchEnvelope.MaxY);
	if (minx > maxx || miny > maxy)
	{
		return 0;//不想交
	}
	else
	{
		return 2;//相交
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