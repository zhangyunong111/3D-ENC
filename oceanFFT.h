#pragma once
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <math_constants.h>

#include <ReadData.h>
#include <rtreeindex.h>
#include <loadobj.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <rendercheck_gl.h>

//#include "GL\glut.h"

#include <png.h>

#include "ogrsf_frmts.h"
#include "ogr_spatialref.h"

const char *sSDKsample = "CUDA FFT Ocean Simulation";

#define MAX_EPSILON 0.10f
#define THRESHOLD   0.15f
#define REFRESH_DELAY     10 //ms

#define MAX_LINE 1024

#define random(a,b) (rand()%(b-a)+a)



////////////////////////////////////////////////////////////////////////////////
// constants
unsigned int windowW = 1200, windowH = 800;//���ڴ�С

const unsigned int meshSize = 1024;
const unsigned int spectrumW = meshSize + 4;
const unsigned int spectrumH = meshSize + 1;

const int frameCompare = 4;

//GLfloat seasize = 1.0f;

// OpenGL vertex buffers
GLuint posVertexBuffer;
GLuint heightVertexBuffer, slopeVertexBuffer;
struct cudaGraphicsResource *cuda_posVB_resource, *cuda_heightVB_resource, *cuda_slopeVB_resource; // handles OpenGL-CUDA exchange

GLuint indexBuffer;
GLuint shaderProg;
char *vertShaderPath = 0, *fragShaderPath = 0;

// mouse controls
int mouseOldX, mouseOldY;
int mouseButtons = 0;
float rotateX = 20.0f, rotateY = 0.0f;
float translateX = 0.0f, translateY = 0.0f, translateZ = -2.0f;
GLdouble scaling = 30.0;//��������
const Point<float > Center = { 118.0f,0.0f,39.0f };//�����е�
const float Center_ratio = 0.2;         //��������

bool animate = true;
bool drawPoints = false;
bool wireFrame = false;
bool g_hasDouble = false;
bool coordinate = true;
bool drawseaPoints = false;
bool drawseaLine = false;
bool drawseaAre = false;

// FFT data
cufftHandle fftPlan;
float2 *d_h0 = 0;   // heightfield at time 0
float2 *h_h0 = 0;
float2 *d_ht = 0;   // heightfield at time t
float2 *d_slope = 0;

// pointers to device object
float *g_hptr = NULL;
float2 *g_sptr = NULL;

// simulation parameters
const float g = 9.81f;              // gravitational constant
const float A = 1e-7f;              // wave scale factor
const float patchSize = 50;        // patch size
float windSpeed = 2.0f;
float windDir = CUDART_PI_F / 3.0f;
float dirDepend = 0.07f;

StopWatchInterface *timer = NULL;
float animTime = 0.0f;
float prevTime = 0.0f;
float animationRate = -0.001f;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

GLuint mTextures[6];//sky����
GLuint bouyTextures[2];//��������

//vector<Features<float>> FeatureIndex;        //Ҫ��
vector<Features<float>> ENCFeatureIndex;        //Ҫ��
//vector<OGRFeature *> ENCFeatureIndex;
int FeatureCount = 0;
OGREnvelope ALLENC_envelope;//������Χ
OGREnvelope Searchenvelope;//��ѯ��Χ

const float pi = 3.14159;
vector<Point<float>> circle;        //Բ
vector<Point<float>> arc;           //Բ��
vector<Point<float>> SOUNDG_P;      //ˮ��
vector<Point<float>> TSSBND_L;      //�ֵ�ͨ�������߽�
vector<Point<float>> RESARE_A;     //������
vector<Point<float>> LNDARE_A;     //½��
vector<Point<float>> LIGHTS_P;     //����
vector<Features<float> *> Qresult[100];//�Ĳ������
vector<Node<float> *> Qnodes[100];
vector<Features<float> *> Rresult[100];//R�����
vector<RTreeNode *> Rnodes[100];
vector<Features<float> *> QRresult[100];//3DENCQRTree���
vector<RTreeNode *> QRnodes[100];
vector<Node<float> *> QRLnodes[100];
vector<Features<float> *> Nresult;//���
//QuadTree *ENC_QuadTree = NULL;

//QuardTree<float> *ENC_Tree = NULL;
//QuardTree<float>* ENC_Tree = new QuardTree<float>();



float4 blackcolor = { 0.0f, 0.0f, 0.0f, 0.5f };//glColor3f(0.0, 0.0, 0.0);  --> ��ɫ
float4 redcolor = { 1.0f, 0.0f, 0.0f, 0.5f };//glColor3f(1.0, 0.0, 0.0);  --> ��ɫ
float4 greencolor = { 0.0f, 1.0f, 0.0f, 0.5f };//glColor3f(0.0, 1.0, 0.0);  --> ��ɫ
float4 bluecolor = { 0.0f, 0.0f, 1.0f, 0.5f };//glColor3f(0.0, 0.0, 1.0);  --> ��ɫ
float4 yellowcolor = { 1.0f, 1.0f, 0.0f, 0.5f };//glColor3f(1.0, 1.0, 0.0);  --> ��ɫ
float4 magentacolor = { 1.0f, 0.0f, 1.0f, 0.5f };//glColor3f(1.0, 0.0, 1.0);  --> Ʒ��ɫ
float4 cyancolor = { 0.0f, 1.0f, 1.0f, 0.5f };//glColor3f(0.0, 1.0, 1.0);  --> ��ɫ
float4 whitecolor = { 1.0f, 1.0f, 1.0f, 0.5f };//glColor3f(1.0, 1.0, 1.0);  --> ��ɫ
float4 color = { 0.5f, 0.5f, 0.0f, 0.5f };//glColor3f(0.5, 0.5, 0.0);  --> 
float4 sandybrowncolor = { 1.0f, 0.64f, 0.37f, 0.5f };//glColor4f(1.0f, 0.75f, 0.0f, 0.5);  --> sandybrown





/*
struct TriangleInfo
{
	int m_iVertexNumber;//���ǻ�������Ŀ
	glm::vec3*  m_vVertice = NULL;//���ǻ��������ϵ������
	glm::vec3*  m_vNormal = NULL;//���ǻ��������ϵ�ķ���
	glm::vec2*  m_vTexcoord = NULL;//���ǻ��������ϵ����������
	glm::ivec3*  m_vTriangle = NULL;//���Ǻ������ϵ�������Ӧ���±�
}m_struTriangleInfo;
*/
////////////////////////////////////////////////////////////////////////////////
// kernels
//#include <oceanFFT_kernel.cu>

extern "C"
void cudaGenerateSpectrumKernel(float2 *d_h0,
	float2 *d_ht,
	unsigned int in_width,
	unsigned int out_width,
	unsigned int out_height,
	float animTime,
	float patchSize);

extern "C"
void cudaUpdateHeightmapKernel(float  *d_heightMap,
	float2 *d_ht,
	unsigned int width,
	unsigned int height,
	bool autoTest);

extern "C"
void cudaCalculateSlopeKernel(float *h, float2 *slopeOut,
	unsigned int width, unsigned int height);

////////////////////////////////////////////////////////////////////////////////
// forward declarations
void runAutoTest(int argc, char **argv);
void runGraphicsTest(int argc, char **argv);

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, int size);
void deleteVBO(GLuint *vbo);
void createMeshIndexBuffer(GLuint *id, int w, int h);
void createMeshPositionVBO(GLuint *id, int w, int h);
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName);
void cube(GLfloat size, GLfloat position);
void small_cube(GLfloat x, GLfloat y, GLfloat z, GLfloat position_longitude, GLfloat position_latitudde);
void BouyCube(Point<float> BouyPosition, GLfloat position_longitude, GLfloat position_latitudde);
void Coordinate();//������
void readdata();
void ReadENC(char* lpPath);
void ReadENCData(const char* Filename, vector<Features<float>> &ENCFeatureIndex);

void normalization(vector<Features<float>> &OriginalFeature);//Ҫ�ع�һ��
void normalizationPoint(Point<float> &OriginalPoint);//������һ��
void normalizationPoints(vector<Point<float>> &OriginalPoint);//������һ��
void normalizationEnvelope(OGREnvelope &OriginalEnvelope);//������Χ��һ��

//�������
double calculateArea(OGRLayer *Layer);//����ͼ�����
void layerIntersection(char* lpPath);//ͼ���ཻ

void CalculateAreaResult();//���������
void CalculateArea(vector<Features<float>> &Feature);//����Ҫ��������Χ�����SMBB���
void FeatureIntersection(vector<Features<float>> Feature);//Ҫ���ཻ
double CalculateEnvelopeIntersection(Features<float > &Feature1, Features<float > &Feature2);//����Ҫ���ཻ���
void PrintArea(string FileName, vector<double> &area);//������
double CalculateEnvelopeOne(Features<float> &Feature);//���㵥һҪ��������Χ���
void CalculateSMBB(Features<float > &Feature);//����С��Ӿ���SMBB
Point<float> RotatePonit(Point<float> &pt, Point<double> &center, double theta); // ĳһ��pt��center��תtheta�Ƕ�
void FindRectangle(vector<Point<float>> &vpt, SmallestMinBoundingBox<float> &SMBB);//����Ӿ���
double CalculateSMBBIntersection(Features<float > &Feature1, Features<float > &Feature2);//����Ҫ��SMBB�ཻ���
float GetCross(Point<float>& p1, Point<float>& p2, Point<float>& p);// ���� |p1 p2| X |p1 p|
bool IsPointInRectangle(Point<float> &p, vector<Point<float>> all_shape_point);//�ж�һ�����Ƿ�����һ������
bool segmentsCrossPoint(Point<float> a, Point<float> b, Point<float> c, Point<float> d, Point<float>& CrossPoint);//�����������θ��ߵ��ཻ��
double DistanceTwoPoints(Point<float> V1, Point<float> V2);//��������֮��ľ���
double TriangleArea(Point<float> V1, Point<float> V2, Point<float> V3);//�������������

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void Wheel(int wheel, int direction, int x, int y);
void reshape(int w, int h);
void timerEvent(int value);
void computeFPS();
void CreatMenu();
void ProcessMenu(int value);

// Cuda functionality
void runCuda();
void runCudaTest(char *exec_path);
void generate_h0(float2 *h0);

//sky
void SkyBoxInit();
void SkyBoxDraw();
GLuint CreateTexture2DFromBMP(const char*bmpPath);
unsigned char * LoadFileContent(const char *path, int &filesize);
unsigned char *DecodeBMP(unsigned char*bmpFileData, int&width, int&height);
GLuint createTexture2D(unsigned char *piexlData, int width, int height, GLenum type);

//features Ҫ��

//void MinimumBoundingBox(MinBoundingBox<float> &MBB);
void DrewSMBB(SmallestMinBoundingBox<float> &MBB);
void DrewEnvelope(OGREnvelope &ENCenvelope, float4 color);
void DrawFeatures();
void DrawPoints(vector<Point<float>> &DrawPoint_P, float size, float4 color);
void DrawLine(vector<Point<float>> &DrawPoint_L, bool point, float4 color);
void DrawAre(vector<Point<float>> &DrawPoint_A, float Point_y, bool point, bool top, float4 color);
//Բ��
void drawCircle(float start_angle, float end_angle);
void drawArc();
void getPointOfCircle(float radius, Point<float> center);
void getPointOfArc(float radius, float start_angle, float end_angle, Point<float> center);

//����PNG����
class gl_texture_t {
public:
	GLsizei width;
	GLsizei height;
	GLenum format;
	GLint internalFormat;
	GLuint id;
	GLubyte *texels;
};
/* Texture id for the demo */
gl_texture_t *ReadPNGFromFile(const char *filename);
GLuint loadPNGTexture(const char *filename);
void GetPNGtextureInfo(int color_type, gl_texture_t *texinfo);
