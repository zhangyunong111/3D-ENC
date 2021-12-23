#pragma once
#include <oceanFFT.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

    // check for command line arguments检查命令行参数
    if (checkCmdLineFlag(argc, (const char **)argv, "qatest"))
    {
        animate       = false;
        fpsLimit = frameCheckNumber;
        runAutoTest(argc, argv);
    }
    else
    {
        printf("[%s]\n\n"
               "Left mouse button          - rotate\n"
               "Middle mouse button        - pan\n"
               "Right mouse button         - zoom\n"
               "'w' key                    - toggle wireframe\n", sSDKsample);

        runGraphicsTest(argc, argv);
    }

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run test //not run
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    // Cuda init
    int dev = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Compute capability %d.%d\n", deviceProp.major, deviceProp.minor);

    // create FFT plan
    checkCudaErrors(cufftPlan2d(&fftPlan, meshSize, meshSize, CUFFT_C2C));

    // allocate memory
    int spectrumSize = spectrumW*spectrumH*sizeof(float2);
    checkCudaErrors(cudaMalloc((void **)&d_h0, spectrumSize));
    h_h0 = (float2 *) malloc(spectrumSize);
    generate_h0(h_h0);
    checkCudaErrors(cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice));

    int outputSize =  meshSize*meshSize*sizeof(float2);
    checkCudaErrors(cudaMalloc((void **)&d_ht, outputSize));
    checkCudaErrors(cudaMalloc((void **)&d_slope, outputSize));

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    prevTime = sdkGetTimerValue(&timer);

    runCudaTest(argv[0]);

    checkCudaErrors(cudaFree(d_ht));
    checkCudaErrors(cudaFree(d_slope));
    checkCudaErrors(cudaFree(d_h0));
    checkCudaErrors(cufftDestroy(fftPlan));
    free(h_h0);

    exit(g_TotalErrors==0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
void runGraphicsTest(int argc, char **argv)
{
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("[%s] ", sSDKsample);
    printf("\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        printf("[%s]\n", argv[0]);
        printf("   Does not explicitly support -device=n in OpenGL mode\n");
        printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
        printf(" > %s -device=n -qatest\n", argv[0]);
        printf("exiting...\n");

        exit(EXIT_SUCCESS);
    }

    // First initialize OpenGL context, so we can properly set the GL for CUDA.首先初始化OpenGL环境，这样我们可以为CUDA正确设置GL。
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.为了实现OpenGL/CUDA互操作的最佳性能，这是必要的。
    if (false == initGL(&argc, argv))
    {
        return;
    }

    findCudaDevice(argc, (const char **)argv);

    // create FFT plan
    checkCudaErrors(cufftPlan2d(&fftPlan, meshSize, meshSize, CUFFT_C2C));

    // allocate memory
    int spectrumSize = spectrumW*spectrumH*sizeof(float2);
    checkCudaErrors(cudaMalloc((void **)&d_h0, spectrumSize));
    h_h0 = (float2 *) malloc(spectrumSize);
    generate_h0(h_h0);
    checkCudaErrors(cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice));

    int outputSize =  meshSize*meshSize*sizeof(float2);
    checkCudaErrors(cudaMalloc((void **)&d_ht, outputSize));
    checkCudaErrors(cudaMalloc((void **)&d_slope, outputSize));

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    prevTime = sdkGetTimerValue(&timer);

    // create vertex buffers and register with CUDA创建顶点缓冲区和注册CUDA
    createVBO(&heightVertexBuffer, meshSize*meshSize*sizeof(float));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_heightVB_resource, heightVertexBuffer, cudaGraphicsMapFlagsWriteDiscard));

    createVBO(&slopeVertexBuffer, outputSize);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_slopeVB_resource, slopeVertexBuffer, cudaGraphicsMapFlagsWriteDiscard));

    // create vertex and index buffer for mesh为网格创建顶点和索引缓冲区
    createMeshPositionVBO(&posVertexBuffer, meshSize, meshSize);
    createMeshIndexBuffer(&indexBuffer, meshSize, meshSize);

    runCuda();
	//QuardTree<float>* ENC_Tree = new QuardTree<float>();
	readdata();
	CreatMenu();

    // register callbacks注册回调函数
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
	glutMouseWheelFunc(Wheel);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // start rendering mainloop
    glutMainLoop();

//	delete ENC_Tree;
}

float urand()
{
    return rand() / (float)RAND_MAX;
}

// Generates Gaussian random number with mean 0 and standard deviation 1.生成均值为0，标准差为1的高斯随机数
float gauss()
{
    float u1 = urand();
    float u2 = urand();

    if (u1 < 1e-6f)
    {
        u1 = 1e-6f;
    }

    return sqrtf(-2 * logf(u1)) * cosf(2*CUDART_PI_F * u2);
}

// Phillips spectrum菲利普斯频谱
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// A - constant
float phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend)
{
    float k_squared = Kx * Kx + Ky * Ky;

    if (k_squared == 0.0f)
    {
        return 0.0f;
    }

    // largest possible wave from constant wind of velocity v恒速风可能产生的最大波v
    float L = V * V / g;

    float k_x = Kx / sqrtf(k_squared);
    float k_y = Ky / sqrtf(k_squared);
    float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

    float phillips = A * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

    // filter out waves moving opposite to wind滤掉逆风移动的海浪
    if (w_dot_k < 0.0f)
    {
        phillips *= dir_depend;
    }

    // damp out waves with very small length w << l
    //float w = L / 10000;
    //phillips *= expf(-k_squared * w * w);

    return phillips;
}

// Generate base heightfield in frequency space在频率空间产生基础高度场
void generate_h0(float2 *h0)
{
    for (unsigned int y = 0; y<=meshSize; y++)
    {
        for (unsigned int x = 0; x<=meshSize; x++)
        {
            float kx = (-(int)meshSize / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
            float ky = (-(int)meshSize / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

            float P = sqrtf(phillips(kx, ky, windDir, windSpeed, A, dirDepend));

            if (kx == 0.0f && ky == 0.0f)
            {
                P = 0.0f;
            }

            //float Er = urand()*2.0f-1.0f;
            //float Ei = urand()*2.0f-1.0f;
            float Er = gauss();
            float Ei = gauss();

            float h0_re = Er * P * CUDART_SQRT_HALF_F;
            float h0_im = Ei * P * CUDART_SQRT_HALF_F;

            int i = y*spectrumW+x;
            h0[i].x = h0_re;
            h0[i].y = h0_im;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda kernels
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
    size_t num_bytes;

    // generate wave spectrum in frequency domain在频域产生波谱
    cudaGenerateSpectrumKernel(d_h0, d_ht, spectrumW, meshSize, meshSize, animTime, patchSize);

    // execute inverse FFT to convert to spatial domain执行逆FFT转换到空间域
    checkCudaErrors(cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE));

    // update heightmap values in vertex buffer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_heightVB_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&g_hptr, &num_bytes, cuda_heightVB_resource));

    cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize, false);

    // calculate slope for shading
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_slopeVB_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&g_sptr, &num_bytes, cuda_slopeVB_resource));

    cudaCalculateSlopeKernel(g_hptr, g_sptr, meshSize, meshSize);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_heightVB_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_slopeVB_resource, 0));
}

//not run
void runCudaTest(char *exec_path)
{
    checkCudaErrors(cudaMalloc((void **)&g_hptr, meshSize*meshSize*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&g_sptr, meshSize*meshSize*sizeof(float2)));

    // generate wave spectrum in frequency domain在频域产生波谱
    cudaGenerateSpectrumKernel(d_h0, d_ht, spectrumW, meshSize, meshSize, animTime, patchSize);

    // execute inverse FFT to convert to spatial domain执行逆FFT转换到空间域
    checkCudaErrors(cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE));

    // update heightmap values更新heightmap值
    cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize, true);

    {
        float *hptr = (float *)malloc(meshSize*meshSize*sizeof(float));
        cudaMemcpy((void *)hptr, (void *)g_hptr, meshSize*meshSize*sizeof(float), cudaMemcpyDeviceToHost);
        sdkDumpBin((void *)hptr, meshSize*meshSize*sizeof(float), "spatialDomain.bin");

        if (!sdkCompareBin2BinFloat("spatialDomain.bin", "ref_spatialDomain.bin", meshSize*meshSize,
                                    MAX_EPSILON, THRESHOLD, exec_path))
        {
            g_TotalErrors++;
        }

        free(hptr);

    }

    // calculate slope for shading计算遮挡坡度
    cudaCalculateSlopeKernel(g_hptr, g_sptr, meshSize, meshSize);

    {
        float2 *sptr = (float2 *)malloc(meshSize*meshSize*sizeof(float2));
        cudaMemcpy((void *)sptr, (void *)g_sptr, meshSize*meshSize*sizeof(float2), cudaMemcpyDeviceToHost);
        sdkDumpBin(sptr, meshSize*meshSize*sizeof(float2), "slopeShading.bin");

        if (!sdkCompareBin2BinFloat("slopeShading.bin", "ref_slopeShading.bin", meshSize*meshSize*2,
                                    MAX_EPSILON, THRESHOLD, exec_path))
        {
            g_TotalErrors++;
        }

        free(sptr);
    }

    checkCudaErrors(cudaFree(g_hptr));
    checkCudaErrors(cudaFree(g_sptr));
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit) {
        fpsCount = 0;
    }
	//printf("FPS %d\n", fpsCount);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // run CUDA kernel to generate vertex positions运行CUDA内核生成顶点位置
    if (animate)
    {
        runCuda();
    }
	clock_t t1 = clock();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix设置视图矩阵
    glMatrixMode(GL_MODELVIEW);//GL_MODELVIEW 模型视图 GL_PROJECTION 投影 GL_TEXTURE 纹理
    glLoadIdentity();//对当前矩阵进行初始化
    glTranslatef(translateX, translateY, translateZ);
    glRotatef(rotateX, 1.0, 0.0, 0.0);
    glRotatef(rotateY, 0.0, 1.0, 0.0);
	//sky
	SkyBoxDraw();
	clock_t t2 = clock();
    
	// render from the vbo从vbo渲染
    glBindBuffer(GL_ARRAY_BUFFER, posVertexBuffer);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, heightVertexBuffer);
    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(1, GL_FLOAT, 0, 0);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, slopeVertexBuffer);
    glClientActiveTexture(GL_TEXTURE1);
    glTexCoordPointer(2, GL_FLOAT, 0, 0);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glUseProgram(shaderProg);

    // Set default uniform variables parameters for the vertex shader为顶点着色器设置默认的统一变量参数
    GLuint uniHeightScale, uniChopiness, uniSize;

    uniHeightScale = glGetUniformLocation(shaderProg, "heightScale");
    glUniform1f(uniHeightScale, 0.5f);

    uniChopiness   = glGetUniformLocation(shaderProg, "chopiness");
    glUniform1f(uniChopiness, 1.0f);

    uniSize        = glGetUniformLocation(shaderProg, "size");
    glUniform2f(uniSize, (float) meshSize, (float) meshSize);

    // Set default uniform variables parameters for the pixel shader为像素着色器设置默认的统一变量参数
    GLuint uniDeepColor, uniShallowColor, uniSkyColor, uniLightDir;

    uniDeepColor = glGetUniformLocation(shaderProg, "deepColor");
    glUniform4f(uniDeepColor, 0.0f, 0.1f, 0.4f, 1.0f);

    uniShallowColor = glGetUniformLocation(shaderProg, "shallowColor");
    glUniform4f(uniShallowColor, 0.1f, 0.3f, 0.3f, 1.0f);

    uniSkyColor = glGetUniformLocation(shaderProg, "skyColor");
    glUniform4f(uniSkyColor, 1.0f, 1.0f, 1.0f, 1.0f);

    uniLightDir = glGetUniformLocation(shaderProg, "lightDir");
    glUniform3f(uniLightDir, 0.0f, 1.0f, 0.0f);
    // end of uniform settings
	clock_t t3 = clock();
    glColor3f(1.0, 1.0, 1.0);

    if (drawPoints)
    {
        glDrawArrays(GL_POINTS, 0, meshSize * meshSize);
    }
    else
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

        glPolygonMode(GL_FRONT_AND_BACK, wireFrame ? GL_LINE : GL_FILL);
        glDrawElements(GL_TRIANGLE_STRIP, ((meshSize*2)+2)*(meshSize-1), GL_UNSIGNED_INT, 0);//使用count个元素定义一个几何序列，这些元素的索引值保存在indices数组中。
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

	//printf("%d\n", ((meshSize * 2) + 2)*(meshSize - 1));
	glDisableClientState(GL_VERTEX_ARRAY);
    glClientActiveTexture(GL_TEXTURE0);//选择当前的纹理单位，以便用顶点数组指定纹理坐标数据，texUnit与glActiveTexture()参数相同
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE1);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    glUseProgram(0);//使用程序对象program作为当前渲染状态的一部分。
	clock_t t4 = clock();
	Coordinate();//绘制坐标轴
	
	static clock_t  sum = 0;
	//clock_t a;
	//a = clock();
	DrawFeatures();//要素
	static clock_t dwTimeLast = clock();
	static int sFPS = 0;
	if (clock() - dwTimeLast >  1000)
	{
		if (sFPS == 0) sFPS = 1;
		printf("FPS = %d tpf = %f\n", sFPS, sum * 1.0 / sFPS);
		sum = 0;
		sFPS = 0;
		dwTimeLast = clock();
	}
	sFPS++;
	clock_t t5 = clock();
	//printf("%ld\n", dwTimeLast);
	//printf("%ld %ld %ld %ld\n", (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000, (t5 - t4) * 1000);
	//cube(seasize);
	//small_cube(0.05, 0.05, 0.05, -1.0, -0.5);

/*	Point<float> center = { 0.01f,0.01f,0.01f };
	BouyCube(center, -0.1f, -0.1f);//浮标
	Node<float> Center;//场景中点
	MaxMBB(FeatureIndex, Center.MBB);
	MinimumBoundingBox(Center.MBB);*/

    glutSwapBuffers();

    computeFPS();
}

void timerEvent(int value)
{
    float time = sdkGetTimerValue(&timer);

    if (animate)
    {
        animTime += (time - prevTime) * animationRate;
    }

    glutPostRedisplay();
    prevTime = time;

    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_heightVB_resource));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_slopeVB_resource));

    deleteVBO(&posVertexBuffer);
    deleteVBO(&heightVertexBuffer);
    deleteVBO(&slopeVertexBuffer);

    checkCudaErrors(cudaFree(d_h0));
    checkCudaErrors(cudaFree(d_slope));
    checkCudaErrors(cudaFree(d_ht));
    free(h_h0);
    cufftDestroy(fftPlan);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouseOldX);
	dy = (float)(y - mouseOldY);

    switch (key)
    {
        case (27) :
            cleanup();
            exit(EXIT_SUCCESS);

        case 'w':
            wireFrame = !wireFrame;
            break;

        case 'p':
            drawPoints = !drawPoints;
            break;

        case ' ':
            animate = !animate;
            break;

		case 'c':
			coordinate = !coordinate;
			break;
/*
		case 's':
			drawseaPoints = !drawseaPoints;
			break;

		case 'd':
			drawseaLine = !drawseaLine;
			break;

		case 'f':
			drawseaAre = !drawseaAre;
			break;

		case '4':
			translateX += 0.01f;
			break;
		case '6':
			translateX -= 0.01f;
			break;
		case '2':
			translateY += 0.01f;
			break;
		case '8':
			translateY -= 0.01f;
			break;
		case 'k':
			translateZ += dy * 0.01f;
			break;*/
		case 'r':
			rotateX += dy * 0.2f;
			rotateY += dx * 0.2f;
			break;
    }

	mouseOldX = x;
	mouseOldY = y;

}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouseButtons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouseButtons = 0;
    }
    mouseOldX = x;
    mouseOldY = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouseOldX);
    dy = (float)(y - mouseOldY);

    if (mouseButtons == 2)
    {
        rotateX += dy * 0.2f;
        rotateY += dx * 0.2f;
    }
    else if (mouseButtons == 1)
    {
        translateX += dx * 0.001f;
        translateY -= dy * 0.001f;
    }
/*   else if (mouseButtons == 4)
    {
        translateZ += dy * 0.01f;
    }
*/ 
    mouseOldX = x;
    mouseOldY = y;
}

//滚轮控制
void Wheel(int wheel, int direction, int x, int y)
{
	if (direction > 0)
	{
		translateZ += 0.1f; 
	}
	else if (direction < 0)
	{
		translateZ -= 0.1f;
	}
}

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(scaling, (double)w / (double)h, 0.0, 10.0);
	


    windowW = w;
    windowH = h;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(windowW, windowH);
    glutCreateWindow("CUDA FFT Ocean Simulation");

	SkyBoxInit();

    vertShaderPath = sdkFindFilePath("ocean.vert", argv[0]);
    fragShaderPath = sdkFindFilePath("ocean.frag", argv[0]);

    if (vertShaderPath == NULL || fragShaderPath == NULL)
    {
        fprintf(stderr, "Error unable to find GLSL vertex and fragment shaders!\n");
        exit(EXIT_FAILURE);
    }

    // initialize necessary OpenGL extensions

    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    if (!areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    // load shader
    shaderProg = loadGLSLProgram(vertShaderPath, fragShaderPath);

    SDK_CHECK_ERROR_GL();
    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, int size)
{
    // create buffer object创建缓冲区对象
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo)
{
    glDeleteBuffers(1, vbo);
    *vbo = 0;
}

// create index buffer for rendering quad mesh创建索引缓冲区渲染四网格
void createMeshIndexBuffer(GLuint *id, int w, int h)
{
    int size = ((w*2)+2)*(h-1)*sizeof(GLuint);

    // create index buffer
    glGenBuffers(1, id);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

    // fill with indices for rendering mesh as triangle strips
    GLuint *indices = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

    if (!indices)
    {
        return;
    }

    for (int y=0; y<h-1; y++)
    {
        for (int x=0; x<w; x++)
        {
            *indices++ = y*w+x;
            *indices++ = (y+1)*w+x;
        }

        // start new strip with degenerate triangle开始新的带退化三角形
        *indices++ = (y+1)*w+(w-1);
        *indices++ = (y+1)*w;
    }

    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// create fixed vertex buffer to store mesh vertices创建固定的顶点缓冲区来存储网格顶点
void createMeshPositionVBO(GLuint *id, int w, int h)
{
    createVBO(id, w*h*4*sizeof(float));

    glBindBuffer(GL_ARRAY_BUFFER, *id);
    float *pos = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

    if (!pos)
    {
        return;
    }

    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            float u = x / (float)(w-1);
            float v = y / (float)(h-1);
            *pos++ = u*2.0f-1.0f;
            *pos++ = 0.0f;
            *pos++ = v*2.0f-1.0f;
            *pos++ = 1.0f;
        }
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Attach shader to a program附加着色器到程序
int attachShader(GLuint prg, GLenum type, const char *name)
{
    GLuint shader;
    FILE *fp;
    int size, compiled;
    char *src;

    fp = fopen(name, "rb");

    if (!fp)
    {
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    src = (char *)malloc(size);

    fseek(fp, 0, SEEK_SET);
    fread(src, sizeof(char), size, fp);
    fclose(fp);

    shader = glCreateShader(type);
    glShaderSource(shader, 1, (const char **)&src, (const GLint *)&size);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, (GLint *)&compiled);

    if (!compiled)
    {
        char log[2048];
        int len;

        glGetShaderInfoLog(shader, 2048, (GLsizei *)&len, log);
        printf("Info log: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }

    free(src);

    glAttachShader(prg, shader);
    glDeleteShader(shader);

    return 1;
}

// Create shader program from vertex shader and fragment shader files从顶点着色和片段着色文件创建着色程序
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName)
{
    GLint linked;
    GLuint program;

    program = glCreateProgram();//创建一个program对象

    if (!attachShader(program, GL_VERTEX_SHADER, vertFileName))
    {
        glDeleteProgram(program);
        fprintf(stderr, "Couldn't attach vertex shader from file %s\n", vertFileName);
        return 0;
    }

    if (!attachShader(program, GL_FRAGMENT_SHADER, fragFileName))
    {
        glDeleteProgram(program);
        fprintf(stderr, "Couldn't attach fragment shader from file %s\n", fragFileName);
        return 0;
    }

    glLinkProgram(program); //连接一个program对象
    glGetProgramiv(program, GL_LINK_STATUS, &linked);//从program对象返回一个参数的值。GL_LINK_STATUS:如果program的最后一个链接操作成功，则params返回GL_TRUE，否则返回GL_FALSE。

    if (!linked)
    {
        glDeleteProgram(program);
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);//返回program对象的信息日志
        fprintf(stderr, "Failed to link program: %s\n", temp);
        return 0;
    }

    return program;
}

void readdata()
{	
	//查询范围
	Searchenvelope.MaxX = 118.2;
	Searchenvelope.MinX = 117.8;
	Searchenvelope.MaxY = 39.2;
	Searchenvelope.MinY = 38.8;

	clock_t t1 = clock();
	//ReadENC("E:\\ArcGIS\\test2\\");
	//ReadENC("E:\\ArcGIS\\test3\\");
	ReadENC("E:\\ArcGIS\\2016_AllENC\\");

	clock_t t2 = clock();
	//创建QTree or 创建3DENCQRTree
	QuadTree *ENC_QTree = NULL;
	ENC_QTree = new QuadTree();
	ENC_QTree->clear();
	int QT_k = 2048;
	int RT_k = 16;
	ENC_QTree->CreatQuad(ALLENC_envelope);

	for (int i = 0; i < ENCFeatureIndex.size(); i++)
	{
		ENC_QTree->Insert(&ENCFeatureIndex[i], QT_k);
		//printf("nodeCount %d \n", ENC_QTree->nodeCount());
	}
	clock_t t3 = clock();
	//ENC_QTree->InsertLeaf(RT_k, false);//不分类插入
	ENC_QTree->InsertLeaf(RT_k, true);//分类插入
	clock_t t4 = clock();
	//创建RTree
	RTreeIndex ENC_RTree;
	for (int i = 0; i < ENCFeatureIndex.size(); i++)
	{
		////printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", ENCFeatureIndex[i].ENCenvelope.MaxX, ENCFeatureIndex[i].ENCenvelope.MaxY, ENCFeatureIndex[i].ENCenvelope.MinX, ENCFeatureIndex[i].ENCenvelope.MinY);

		ENC_RTree.Insert(&ENCFeatureIndex[i], ALLENC_envelope, RT_k);

		//printf("节点数 %d \n", ENC_RTree.root->Count());
		//printf("节点数 %d \n", ENC_RTree.root->);
	}
	//printf("节点数 %d %d \n", mm,ENC_RTree.root->Count());
	
	clock_t t5 = clock();
/*	RTreeIndex ENC_RTree[3];
	int number[3] = {0,0,0};
	int mm, nn;
	for (int i = 1; i < ENCFeatureIndex.size(); i++)
	{
		if (ENCFeatureIndex[i].FeatureType ==0)
		{
			ENC_RTree[0].Insert(&ENCFeatureIndex[i]);
			number[0]++;
		}
		if (ENCFeatureIndex[i].FeatureType == 1)
		{
			ENC_RTree[1].Insert(&ENCFeatureIndex[i]);
			number[1]++;
		}
		if (ENCFeatureIndex[i].FeatureType == 2)
		{
			ENC_RTree[2].Insert(&ENCFeatureIndex[i]);
			number[2]++;
		}
	}
	printf("环境要素节点数：%d 实体要素节点数：%d 虚拟要素节点数：%d\n", number[0], number[1], number[2]);
	clock_t t4 = clock();*/
	// R树查询测试
	for (int i = 0; i < 100; i++)
	{
		ENC_RTree.root->QRTreeSearch(Searchenvelope, Rresult[i], Rnodes[i]);
	}
	clock_t t6 = clock();
	//3DENCQRTree查询测试
	for (int i = 0; i < 100; i++)
	{
		ENC_QTree->RegionSearch(Searchenvelope, QRresult[i], QRnodes[i], QRLnodes[i]);
	}
	////ENC_QTree->RegionResearch(envelope.MinX, envelope.MaxX, envelope.MinY, envelope.MaxY, visitednum, foundnum);

	clock_t t7 = clock();
	// QTree查询测试
	for (int i = 0; i < 100; i++)
	{
		ENC_QTree->QTreeResearch(Searchenvelope, Qresult[i], Qnodes[i]);
	}

	clock_t t8 = clock();

	for (int i = 0; i < ENCFeatureIndex.size(); i++)
	{
		if (Searchenvelope.Intersects(ENCFeatureIndex[i].ENCenvelope))
		{
			for (int t = 0; t < ENCFeatureIndex[i].point.size(); t++)
			{
				int m;
				m++;
			}
			Nresult.push_back(&ENCFeatureIndex[i]);
		}
	}
	clock_t t9 = clock();
	//printf("mm %d \n", mm);
	printf("nodeCount %d \n", ENC_QTree->nodeCount());
	printf("加载数据时间 %ld ms\n", t2 - t1);
	printf("创建3DENCQRTree时间 %ld ms\n", t4 - t2);
	printf("创建Q树时间 %ld ms\n", t3 - t2);
	printf("创建R树时间 %ld ms\n", t5 - t4);
	printf("R树查询时间 %ld ms\n", t6 - t5);
	printf("3DENCQRTree查询时间 %ld ms\n", t7 - t6);
	printf("Q树查询时间 %ld ms\n", t8 - t7);
	printf("N查询时间 %ld ms\n", t9 - t8);
	printf("R树查询节点数 %d \n", Rresult[0].size());
	printf("R树查询节点数 %d \n", Rnodes[0].size());
	printf("3DENCQRTree查询节点数 %d \n", QRresult[0].size());
	//printf("3DENCQRTree查询节点数 %d \n", QRnodes.size());
	printf("Q树查询节点数 %d \n", Qresult[0].size());
	printf("N查询节点数 %d \n", Nresult.size());
	//printf("visitednum：%d foundnum：%d \n", visitednum, foundnum);
	/*
	for (int i = 0; i < Rresult.size(); i++)
	{
		printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", Rresult[i]->ENCenvelope.MaxX, Rresult[i]->ENCenvelope.MaxY, Rresult[i]->ENCenvelope.MinX, Rresult[i]->ENCenvelope.MinY);
	}
	for (int i = 0; i < Qresult.size(); i++)
	{
		int k = -1;
		for (int j = 0; j < QRresult.size(); j++)
		{
			if (QRresult[i] == Qresult[j])
			{
				k = j;
			}
		}
		if (k = -1)
		{
			printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", Qresult[i]->ENCenvelope.MaxX, Qresult[i]->ENCenvelope.MaxY, Qresult[i]->ENCenvelope.MinX, Qresult[i]->ENCenvelope.MinY);
		}
	}*/
	int temp[9] = {0,0,0,0,0,0,0,0,0};
	for (int i = 0; i < Rresult[0].size(); i++)
	{
		if (Rresult[0][i]->FeatureType == 0) temp[0] += 1;
		if (Rresult[0][i]->FeatureType == 1) temp[1] += 1;
		if (Rresult[0][i]->FeatureType == 2) temp[2] += 1;
	}	
	for (int i = 0; i < QRresult[0].size(); i++)
	{
		if (QRresult[0][i]->FeatureType == 0) temp[3] += 1;
		if (QRresult[0][i]->FeatureType == 1) temp[4] += 1;
		if (QRresult[0][i]->FeatureType == 2) temp[5] += 1;
	}
	for (int i = 0; i < Qresult[0].size(); i++)
	{
		if (Qresult[0][i]->FeatureType == 0) temp[6] += 1;
		if (Qresult[0][i]->FeatureType == 1) temp[7] += 1;
		if (Qresult[0][i]->FeatureType == 2) temp[8] += 1;
	}
	printf("环境要素：%d 实体要素：%d 虚拟要素：%d \n", temp[0], temp[1], temp[2]);
	printf("环境要素：%d 实体要素：%d 虚拟要素：%d \n", temp[3], temp[4], temp[5]);
	printf("环境要素：%d 实体要素：%d 虚拟要素：%d \n", temp[6], temp[7], temp[8]);

	normalization(ENCFeatureIndex);//归一化
	normalizationEnvelope(ALLENC_envelope);
	normalizationEnvelope(Searchenvelope);
	for (int i = 0; i < Qnodes[0].size(); i++)
	{
		normalizationEnvelope(Qnodes[0][i]->ENCenvelope);
	}
	for (int i = 0; i < Nresult.size(); i++)
	{
		normalizationEnvelope(Nresult[i]->ENCenvelope);
	}
	for (int i = 0; i < Rnodes[0].size(); i++)
	{
		//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", Rnodes[0][i]->envelope.MaxX, Rnodes[0][i]->envelope.MaxY, Rnodes[0][i]->envelope.MinX, Rnodes[0][i]->envelope.MinY);
		
		normalizationEnvelope(Rnodes[0][i]->envelope);
	}
	for (int i = 0; i < QRnodes[0].size(); i++)
	{
		//normalizationEnvelope(QRnodes[0][i]->envelope);
	}
	

/*	//FindData("E:/program/oceanFFT", fileList);
								  //输出文件夹下所有文件的名称 
	for (int i = 0; i < fileList.size(); i++)
	{
		string str = fileList[i];
	    const char *shapename = str.data();
	    printf(shapename);
//		ENC_QuadTree->readpointdata(shapename, SOUNDG_P);
		cout << fileList[i] << endl;
	}
	cout << "文件数目：" << fileList.size() << endl;


	//ENC_QuadTree->readpointdata("SOUNDG_P.txt", SOUNDG_P);
	ENC_QuadTree->readlinedata("TSSbND_L.txt", TSSBND_L);
	ENC_QuadTree->readaredata("RESARE_A.txt", RESARE_A);
	ENC_QuadTree->readaredata("LNDARE_A.txt", LNDARE_A);
	//ENC_QuadTree->readpointdata("LIGHTS_P.txt", LIGHTS_P);
	//ENC_QuadTree->readshape();
	//ENC_QuadTree->Insert(SOUNDG_P);
	//ENC_QuadTree->Insert;
	//printf("%d\n", ENC_QuadTree->nodeCount());
	Circle *c_c = NULL;
	c_c = new Circle();
	c_c->setR(2);
	c_c->show();
	readlinedata("TSSbND_L.txt", TSSBND_L);
	readaredata("RESARE_A.txt", RESARE_A);
	readaredata("LNDARE_A.txt", LNDARE_A);
	//delete ENC_Tree;*/
}

//读取海图文件
void ReadENC(char* lpPath)
{
	char szFind[MAX_PATH];
	WIN32_FIND_DATA FindFileData;

	strcpy(szFind, lpPath);
	strcat(szFind, "\\*.000");
	//strcat(szFind, "\\CN323101.000");
	//strcat(szFind, "\\CN321001.000"); 
	//strcat(szFind, "\\CN322001.000"); 
	//strcat(szFind, "\\CN324101.000");
	//strcat(szFind, "\\CN333001.000");
	//strcat(szFind, "\\CN334001.000");

	HANDLE hFind = ::FindFirstFile(szFind, &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)  return;
	int filenum = 0;
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
				ReadENC(szFile);
			}
		}
		//else if (filenum < 42)
		else if (filenum < 32)
		{
			string str = FindFileData.cFileName;
			char ENCdataPath[100];
			strcpy(ENCdataPath, lpPath);
			strcat(ENCdataPath, str.data());			
			printf("文件名：%s\n", str.data());//海图文件名
			ReadENCData(ENCdataPath, ENCFeatureIndex);
		}
		else
		{
			break;
		}
		if (!FindNextFile(hFind, &FindFileData))  break;
		filenum++;
	}

	FindClose(hFind);
}
//读取海图数据
void ReadENCData(const char* Filename, vector<Features<float>> &ENCFeatureIndex)
{
	GDALAllRegister();
	CPLSetConfigOption("GDAL_DATA", "D:\\warmerda_release\\bin\\gdal-data");        //环境变量设置，必须调用
	GDALDataset *poDS;
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");  //解决中文乱码问题
											   
	poDS = (GDALDataset*)GDALOpenEx(Filename, GDAL_OF_VECTOR, NULL, NULL, NULL);//读取海图.000文件

	if (poDS == NULL)
	{
		printf("Open failed.\n%s");
	}
/*	else
	{
		printf("Open successful.\n%s");//打开成功
	}
	printf("驱动名称：%s   图层数：%d \n", poDS->GetDriverName(), poDS->GetLayerCount());//获取驱动名称，图层数
	printf("图层数：%d \n", poDS->GetLayerCount());//获取图层数
	printf("光栅数：%d \n", poDS->GetRasterCount());//
	printf("RasterXSize：%d \n", poDS->GetRasterXSize());//
	printf("RasterYSize：%d \n", poDS->GetRasterYSize());//*/
	
	int PointCount = 0;
	OGRLayer *FirstLayer = poDS->GetLayer(1);
	OGREnvelope ALLenvelope;
	FirstLayer->GetExtent(&ALLenvelope, TRUE);
	ALLENC_envelope.Merge(ALLenvelope);
	printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", ALLENC_envelope.MaxX, ALLENC_envelope.MaxY, ALLENC_envelope.MinX, ALLENC_envelope.MinY);

	//layerIntersection("E:/ArcGIS/CN323101/");

	/*

	for (int i = 0; i < poDS->GetLayerCount(); i++)//读取每个图层
	{
		OGRLayer *Layer1 = poDS->GetLayer(i);
		if (wkbFlatten(Layer1->GetGeomType()) == 3)
		{
			for (int j = i + 1; j < poDS->GetLayerCount(); j++)
			{
				OGRLayer *Layer2 = poDS->GetLayer(j);
				if (wkbFlatten(Layer2->GetGeomType()) == 3)
				{
					const char *pszName = "NewShape";
					//GDALDriver *poDriver =GDALCreateDriver(); 
					GDALDataset *poDat; 
					//poDat = (GDALDataset*)GDALCreate(pszDriverName, );
					//(GDALDataset*)GDALCreateDatasetMaskBand(poDriver,NULL);  
					GDALDriverManager;
					if (poDat == NULL)
					{
						printf("%s driver not available.\n", pszName);
						exit(1);
					}
					GDALDriver* poDriver = (GDALDriver *)GDALGetDriverByName("ESRI Shapefile");
					GDALDataset *pDstDS = poDriver->Create(pszName, 1000, 1000, 2, GDT_Byte, NULL);

					GDALDataset *pPolygonDS;
					GDALDriver  *pDriverPng = (GDALDriver *)GDALGetDriverByName("ESRI Shapefile");
					pPolygonDS = pDriverPng->CreateCopy(pszName, pDstDS, FALSE, NULL, NULL, NULL);
					
					//GDALDriver *poDriver = OGRSFDriverRegistrar::GetRegistrar()->GetDriverByName("ESRI Shapefile");
					//OGRSFDriver *poDriver = GDALDriverManager::GetDriverByName("ESRI Shapefile");
					//GDALDataset *poDstDS = poDriver->;
					////OGRLayer *LayerResult;
					//poDat = poDriver->CreateDataSource("point_out.shp", NULL);
					OGRSpatialReference *pSRS = Layer1->GetSpatialRef();
					//char **papszOptions;
					//papszOptions = CSLSetNameValue(papszOptions, "DIM", "2");
					OGRLayer *LayerResult = pPolygonDS->CreateLayer("Newpolygon", pSRS, wkbPolygon, NULL);
					//OGRLayer *LayerResult = poDriver->CopyLayer(Layer1, "Newpolygon", NULL);
					if (LayerResult==NULL)
					{
						printf("failed\n");
					}
					GDALDataset *Dataset1 = (GDALDataset*)GDALOpenEx("E:/ArcGIS/CN323101/ACHARE_A.shp", GDAL_OF_VECTOR, NULL, NULL, NULL); 
					GDALDataset *Dataset2 = (GDALDataset*)GDALOpenEx("E:/ArcGIS/CN323101/CANALS_A.shp", GDAL_OF_VECTOR, NULL, NULL, NULL);
					OGRLayer *Dataset1Layer = Dataset1->GetLayer(0);
					OGRLayer *Dataset2Layer = Dataset2->GetLayer(0);
					double L11 = calculateArea(Dataset1Layer);
					double L12 = calculateArea(Dataset2Layer);
					printf("%f ,%f\n", L11, L12);
					Dataset1Layer->Intersection(Dataset2Layer, LayerResult, NULL, GDALTermProgress, NULL);
					/*
					double L1 = calculateArea(Layer1);
					double L2 = calculateArea(Layer2);
					printf("%f ,%f\n", L1, L2);
					Layer1->Intersection(Layer2, LayerResult, NULL, GDALTermProgress, NULL);
					double LR = calculateArea(LayerResult);
					printf("%f\n", LR);
					//printf("%f ,%f ,%f\n", L1, L2, LR);
					GDALClose(pDstDS);
					GDALClose(pPolygonDS);
					pPolygonDS->DeleteLayer(0);
					//Dataset2->DeleteLayer(0);
					
				}
			}
		}
	
	}*/
	for (int i = 0; i < poDS->GetLayerCount(); i++)//读取每个图层
	{
		OGRLayer *poLayer = poDS->GetLayer(i);
		//获取图层名称、要素数、集合类型
		//printf("\n图层%d名称：%s   要素数：%d  图层集合类型：%d\n", i + 1, poLayer->GetName(), poLayer->GetFeatureCount(), wkbFlatten(poLayer->GetGeomType()));
		FeatureCount += poLayer->GetFeatureCount();//要素总数

		poLayer->ResetReading(); //把要素读取顺序重置为第一个开始
		int featuretype = LayerType(poLayer->GetName());//判断要素分类类型
		OGRFeature *poFeature;

		while ((poFeature = poLayer->GetNextFeature()) != NULL)//获得图层中每个要素
		{
			OGRGeometry *poGeometry;
			poGeometry = poFeature->GetGeometryRef();//获取要素中几何体的指针
			//ENCFeatureIndex.push_back(poFeature);
			
			if (poGeometry == NULL)//非几何要素
			{
				break;
			}

			if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint)
			{
				Features<float> PointFeature;
				PointFeature.ENCFeature = poFeature;
				PointFeature.ENCGeometry = poGeometry;
				OGREnvelope P_envelope;
				poGeometry->getEnvelope(&P_envelope);
				PointFeature.ENCenvelope = P_envelope;
				PointFeature.ENCFeatureDefn = poFeature->GetDefnRef();
				PointFeature.FeatureType = featuretype;

				OGRPoint *poPoint = (OGRPoint *)poGeometry;
				Point<float> p;
				p.x = poPoint->getX();
				p.y = 0;
				p.z= poPoint->getY();
				PointFeature.point.push_back(p);
				//ENC_QTree->Insert(p);
				PointFeature.PointNum = 1;
				PointCount++;
				//printf("%.3f,%.3f\n", poPoint->getX(), poPoint->getY());
				CalculateSMBB(PointFeature);
				ENCFeatureIndex.push_back(PointFeature);
			}
			else if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbLineString)
			{
				Features<float> LineStringFeature;
				LineStringFeature.ENCFeature = poFeature;
				LineStringFeature.ENCGeometry = poGeometry;
				LineStringFeature.ENCFeatureDefn = poFeature->GetDefnRef();
				OGREnvelope P_envelope;
				poGeometry->getEnvelope(&P_envelope);
				LineStringFeature.ENCenvelope = P_envelope;
				LineStringFeature.FeatureType = featuretype;

				OGRLineString *poLine = (OGRLineString *)poGeometry;
				//LineStringFeature.PointNum = poLine->getNumPoints();
				//printf("%d\n", poLine->getNumPoints());
				int NK = poLine->getNumPoints() / 50 + 1;
				int PK = poLine->getNumPoints() / NK + 1;
				for (int j = 0; j < NK; j++)
				{
					Features<float> SubLineStringFeature;
					SubLineStringFeature = LineStringFeature;
					for (int k = j*PK; k < (j+1)*PK; k++)
					{
						if (k == poLine->getNumPoints())
						{
							break;
						}
						Point<float> p;
						p.x = poLine->getX(k);
						p.y = 0;
						p.z = poLine->getY(k);

						SubLineStringFeature.point.push_back(p);
						SubLineStringFeature.PointNum++;
						PointCount++;
					}
					CalculateSMBB(SubLineStringFeature);
					ENCFeatureIndex.push_back(SubLineStringFeature);
				}

				//printf("%.3f,%.3f\n", poLine->getX(0), poLine->getY(0));

			}
/*			else if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbLineString)
			{
				Features<float> LineStringFeature;
				LineStringFeature.ENCFeature = poFeature;
				LineStringFeature.ENCGeometry = poGeometry;
				LineStringFeature.ENCFeatureDefn = poFeature->GetDefnRef();
				OGREnvelope P_envelope;
				poGeometry->getEnvelope(&P_envelope);
				LineStringFeature.ENCenvelope = P_envelope;
				LineStringFeature.FeatureType = featuretype;

				OGRLineString *poLine = (OGRLineString *)poGeometry;
				LineStringFeature.PointNum = poLine->getNumPoints();
				//poLine->getEnvelope(Feature.ENCenvelope);
				for (int k = 0; k < poLine->getNumPoints(); k++)
				{
					Point<float> p;
					p.x = poLine->getX(k);
					p.y = 0;
					p.z = poLine->getY(k);
					LineStringFeature.point.push_back(p);
					PointCount++;
					//LineStringFeature.ENCenvelope = P_envelope;
				}
				
				//printf("%.3f,%.3f\n", poLine->getX(0), poLine->getY(0));
				CalculateSMBB(LineStringFeature);
				ENCFeatureIndex.push_back(LineStringFeature);
			}*/
			else if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon)
			{
				Features<float> PolygonFeature;
				PolygonFeature.ENCFeature = poFeature;
				PolygonFeature.ENCGeometry = poGeometry;
				OGREnvelope P_envelope;
				poGeometry->getEnvelope(&P_envelope);
				PolygonFeature.ENCenvelope = P_envelope;
				PolygonFeature.ENCFeatureDefn = poFeature->GetDefnRef();
				PolygonFeature.FeatureType = featuretype;

				//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", PolygonFeature.ENCenvelope->MaxX, PolygonFeature.ENCenvelope->MaxY, PolygonFeature.ENCenvelope->MinX, PolygonFeature.ENCenvelope->MinY);

				OGRPolygon *poPolygon = (OGRPolygon *)poGeometry;
				
				if (poPolygon->getNumInteriorRings() != 0)//内环
				{
					//printf("内环数：%d\n", poPolygon->getNumInteriorRings());
					for (int k = 0; k < poPolygon->getNumInteriorRings(); k++)
					{
						OGRLinearRing *poInteriorRing = poPolygon->getInteriorRing(k);
						PolygonFeature.PointNum += poInteriorRing->getNumPoints();
						for (int m = 0; m < poInteriorRing->getNumPoints(); m++)
						{
							Point<float> p;
							p.x = poInteriorRing->getX(k);
							p.y = 0;
							p.z = poInteriorRing->getY(k);
							//printf("%.3f,%.3f\n", poInteriorRing->getX(k), poInteriorRing->getY(m));
							PolygonFeature.point.push_back(p);
							PointCount++;
						}
					}
				}

				OGRLinearRing *poExteriorRing = poPolygon->getExteriorRing();
				PolygonFeature.PointNum += poExteriorRing->getNumPoints();
				for (int k = 0; k < poExteriorRing->getNumPoints(); k++)//外环
				{
					Point<float> p;
					p.x = poExteriorRing->getX(k);
					p.y = 0;
					p.z = poExteriorRing->getY(k);
					//printf("%.3f,%.3f\n", poExteriorRing->getX(k), poExteriorRing->getY(k));
					PolygonFeature.point.push_back(p);
					PointCount++;
				}
				CalculateSMBB(PolygonFeature);
				ENCFeatureIndex.push_back(PolygonFeature);
			}
			else if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPoint)
			{
				OGRMultiPoint *MultiPoint = (OGRMultiPoint *)poGeometry;
				for (int k = 0; k < MultiPoint->getNumGeometries(); k++)
				{
					Features<float> MultiPointFeature;
					MultiPointFeature.ENCFeature = poFeature;
					MultiPointFeature.ENCGeometry = poGeometry;
					MultiPointFeature.ENCFeatureDefn = poFeature->GetDefnRef();
					MultiPointFeature.FeatureType = featuretype;

					OGRGeometry * SubMultiPoint = MultiPoint->getGeometryRef(k);
					OGRPoint *SubPoint = (OGRPoint *)SubMultiPoint;
					OGREnvelope MultiP_envelope;
					SubMultiPoint->getEnvelope(&MultiP_envelope);
					MultiPointFeature.ENCenvelope = MultiP_envelope;
					Point<float> p;
					p.x = SubPoint->getX();
					p.y = 0;
					p.z = SubPoint->getY();
					MultiPointFeature.point.push_back(p);
					MultiPointFeature.PointNum = 1;
					//printf("%.3f,%.3f\n", p.x, p.z);
					CalculateSMBB(MultiPointFeature);
					ENCFeatureIndex.push_back(MultiPointFeature);
					PointCount++;
				}
			}
			else
			{
				printf("no point geometry\n");
			}
		}
		//printf("叶子节点数：%d   %d", ENC_RTree.root->Count(),k);
		OGRFeature::DestroyFeature(poFeature);

	}
	//CalculateAreaResult();//输出计算结果
	printf("点要素数 %d\n", PointCount);//获取要素总数
	printf("几何要素总数 %d\n", ENCFeatureIndex.size());//获取几何要素总数
	printf("要素总数：%d\n", FeatureCount);//获取要素总数
	GDALClose(poDS);

/*	
	while ((poFeature = poLayer->GetNextFeature()) != NULL)//获得要素，本实例指的是五个点，所以会循环5次
	{
		OGRFeatureDefn *poFDefn = poLayer->GetLayerDefn();
		int iField;
		int i = poFDefn->GetFieldCount(); //获得字段的数目，本实例返回5，不包括前两个字段（FID,Shape），这两个字段在arcgis里也不能被修改;
		for (iField = 0; iField < poFDefn->GetFieldCount(); iField++)
		{
			OGRFieldDefn *poFieldDefn = poFDefn->GetFieldDefn(iField);
			//根据字段值得类型，选择对应的输出
			if (poFieldDefn->GetType() == OFTInteger)
				printf("%d,", poFeature->GetFieldAsInteger(iField));
			else if (poFieldDefn->GetType() == OFTReal)
				printf("%.3f,", poFeature->GetFieldAsDouble(iField));
			else if (poFieldDefn->GetType() == OFTString)
				printf("%s,", poFeature->GetFieldAsString(iField));
			else
				printf("%s,", poFeature->GetFieldAsString(iField));
		}
	}
	*/

}
//要素归一化
void normalization(vector<Features<float>> &OriginalFeature)
{
	for (int i = 0; i < OriginalFeature.size(); i++)
	{
		for (int j = 0; j < OriginalFeature[i].point.size(); j++)
		{
			normalizationPoint(OriginalFeature[i].point[j]);//要素点归一化
		}
		for (int j = 0; j < 4; j++)
		{
			normalizationPoint(OriginalFeature[i].SMBB.p[j]);//SMBB归一化
		}

		if (OriginalFeature[i].ENCenvelope.MaxX != NULL)
		{
			normalizationEnvelope(OriginalFeature[i].ENCenvelope);//四至范围归一化
		}
	}
}
//坐标点归一化
void normalizationPoint(Point<float> &OriginalPoint)
{
	OriginalPoint.x = (OriginalPoint.x - Center.x) / Center_ratio;
	OriginalPoint.y = 0.0f;
	OriginalPoint.z = -(OriginalPoint.z - Center.z) / Center_ratio;//z轴取负数，将数据坐标翻转，匹配视坐标系
}
//坐标点归一化
void normalizationPoints(vector<Point<float>> &OriginalPoint)
{
	for (int i = 0; i < OriginalPoint.size(); i++)
	{
		normalizationPoint(OriginalPoint[i]);
	}
}
//四至范围归一化
void normalizationEnvelope(OGREnvelope &OriginalEnvelope)
{
	OriginalEnvelope.MaxX = (OriginalEnvelope.MaxX - Center.x) / Center_ratio;
	OriginalEnvelope.MinX = (OriginalEnvelope.MinX - Center.x) / Center_ratio;
	OriginalEnvelope.MaxY = -(OriginalEnvelope.MaxY - Center.z) / Center_ratio;//z轴取负数，将数据坐标翻转，匹配视坐标系
	OriginalEnvelope.MinY = -(OriginalEnvelope.MinY - Center.z) / Center_ratio;
}

//图层相交
void layerIntersection(char* lpPath)
{
	char szFind[MAX_PATH];
	WIN32_FIND_DATA FindFileData;

	strcpy(szFind, lpPath);
	strcat(szFind, "\\*.shp");

	HANDLE hFind = ::FindFirstFile(szFind, &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)  return;

	GDALDataset *Dataset;
	vector<OGRLayer *> LayerVector;

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
				layerIntersection(szFile);
			}
		}
		else
		{
			string str = FindFileData.cFileName;
			const char *shapename = str.data();

			for (int j = 0; shapename[j] != '\0'; j++)
			{
				char shapedataPath[100];
				strcpy(shapedataPath, lpPath);
				strcat(shapedataPath, shapename);

				if (shapename[j] == '_' && shapename[j + 1] == 'P')
				{
					break;
				}
				if (shapename[j] == '_' && shapename[j + 1] == 'L')
				{
					break;
				}
				if (shapename[j] == '_' && shapename[j + 1] == 'A')
				{
					GDALDataset *Dataset = (GDALDataset*)GDALOpenEx(shapedataPath, GDAL_OF_VECTOR, NULL, NULL, NULL);
					OGRLayer *DatasetLayer = Dataset->GetLayer(0);
					LayerVector.push_back(DatasetLayer);
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
	GDALDriver* poDriver = (GDALDriver *)GDALGetDriverByName("ESRI Shapefile");
	GDALDataset *pDstDS = poDriver->Create("NewShape", 1000, 1000, 1, GDT_Byte, NULL);

	GDALDataset *pPolygonDS;
	GDALDriver  *pDriverPng = (GDALDriver *)GDALGetDriverByName("ESRI Shapefile");
	pPolygonDS = pDriverPng->CreateCopy("NewShape", pDstDS, FALSE, NULL, NULL, NULL);

	for (int i = 0; i < LayerVector.size(); i++)
	{
		OGRLayer *DatasetLayer1 = LayerVector[i];
		double L11 = calculateArea(DatasetLayer1);
		for (int j = i+1; j < LayerVector.size(); j++)
		{
			OGRLayer *DatasetLayer2 = LayerVector[j];
			double L12 = calculateArea(DatasetLayer2);
			printf("%f ,%f\n", L11, L12);
			char LayerName[10];
			strcpy(LayerName, "New");
			sprintf(LayerName, "%d", i);

			OGRSpatialReference *pSRS = DatasetLayer1->GetSpatialRef();
			OGRLayer *LayerResult = pPolygonDS->CreateLayer(LayerName, pSRS, wkbPolygon, NULL);
			DatasetLayer1->Intersection(DatasetLayer2, LayerResult, NULL, GDALTermProgress, NULL);
			double L13 = calculateArea(LayerResult);
			printf("%f\n", L13);
			pPolygonDS->DeleteLayer(0);
		}
	}
	GDALClose(Dataset);
	GDALClose(pDstDS);
	GDALClose(pPolygonDS);
}
//计算图层面积
double calculateArea(OGRLayer *Layer)
{
	double Area;
	OGRFeature *Feature;
	while ((Feature = Layer->GetNextFeature()) != NULL)//获得图层中每个要素
	{
		OGRGeometry *poGeometry;
		poGeometry = Feature->GetGeometryRef();//获取要素中几何体的指针
		if (poGeometry == NULL)//非几何要素
		{
			break;
		}
		if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon)
		{
			OGRPolygon *poPolygon = (OGRPolygon *)poGeometry;
			OGRSpatialReference p_spRef;

			p_spRef.SetGeogCS("Mygeographic coordinate system",
				"Popular_Visualisation_Datum",
				"My WGS84 Spheroid",
				SRS_WGS84_SEMIMAJOR, SRS_WGS84_INVFLATTENING,
				"Greenwich", 0.0,
				"metre");
			p_spRef.SetProjection("Mercator_1SP");
			poPolygon->transformTo(&p_spRef);
			Area += poPolygon->get_Area();
			//printf("多边形面积：%f\n", poPolygon->get_Area());
		}
	}
	return Area;
}

//计算结果
void CalculateAreaResult() 
{	
	CalculateArea(ENCFeatureIndex);
	FeatureIntersection(ENCFeatureIndex);
	/*
	string FileName;
	ofstream outfile;

	outfile.open(FileName, ostream::app);
	outfile << "----------------------------one frame-------------------------" << endl;

	outfile << "FeatureNumber:" << area[i] << endl;
	outfile << "FeatureEnvelopeArea:" << area[i] << endl;
	outfile << "FeatureEnvelopeArea:" << area[i] << endl;
	outfile << "FeatureEnvelopeArea:" << area[i] << endl;
	outfile << "FeatureEnvelopeArea:" << area[i] << endl;
	outfile << "FeatureEnvelopeArea:" << area[i] << endl;
	outfile << "FeatureEnvelopeArea:" << area[i] << endl;
	outfile << "FeatureEnvelopeArea:" << area[i] << endl;
	outfile << "FeatureSMBBArea:" << area[i] << endl;

	outfile.close();*/
}
//计算要素四至范围面积与SMBB面积
void CalculateArea(vector<Features<float>> &Feature)
{
	vector<double> S_Area, F_Area;
	for (int i = 0; i < Feature.size(); i++)
	{
		double temp1, temp2;
		if (Feature[i].SMBB.area != NULL)
		{
			//printf("MaxX:%.3f ,MaxY:%.3f ,MinY:%.3f \n", ENCFeatureIndex[i].SMBB.p[0].x, ENCFeatureIndex[i].SMBB.p[0].z, ENCFeatureIndex[i].SMBB.area);
			temp1 = Feature[i].SMBB.area * 1000;
			//printf("%.3f \n", temp);
			S_Area.push_back(temp1);
		}
		else
		{
			temp1 = 0.0f;
			S_Area.push_back(temp1);
		}

		if (Feature[i].ENCenvelope.MaxX != NULL)
		{
			temp2 = CalculateEnvelopeOne(Feature[i]) * 1000;
			F_Area.push_back(temp2);
		}
		else
		{
			temp2 = 0.0f;
			F_Area.push_back(temp2);
		}
	}

	//PrintArea("S_Area.txt", S_Area);
	//PrintArea("F_Area.txt", F_Area);
}
//计算要素相交面积与SMBB相交面积
void FeatureIntersection(vector<Features<float>> Feature)
{
	vector<double> FI_Area, SI_Area;
	vector<int> FeatureType1, FeatureType2;
	//for (int i = 6; i < 8; i++)
	for (int i = 1; i < Feature.size() - 28; i++)	
	{			
		if (wkbFlatten(Feature[i].ENCGeometry->getGeometryType()) == wkbPolygon)
		{
			for (int j = i+1; j < Feature.size() - 28; j++)
			{
				if (wkbFlatten(Feature[j].ENCGeometry->getGeometryType()) == wkbPolygon)
				{
					double area1 = CalculateEnvelopeIntersection(Feature[i], Feature[j]) * 1000;
					double area2 = CalculateSMBBIntersection(Feature[i], Feature[j]) * 1000;
					if (area1>0)
					{
						FeatureType1.push_back(Feature[i].FeatureType);
						FeatureType2.push_back(Feature[j].FeatureType);
						FI_Area.push_back(area1);
						SI_Area.push_back(area2);
						//printf("要素1SMBB：%f 要素2SMBB：%f\n", Feature[i].SMBB.area*1000, Feature[j].SMBB.area*1000);
						//printf("相交面积1：%f 要素类型：%d %d\n", area1, Feature[i].FeatureType, Feature[j].FeatureType);
						//printf("相交面积2：%f 要素类型：%d %d\n", area2, Feature[i].FeatureType, Feature[j].FeatureType);
					}
				}
			}
		}
	}
/*	
	double area1 = CalculateEnvelopeIntersection(Feature[6], Feature[7]) * 1000;
	double area2 = CalculateSMBBIntersection(Feature[6], Feature[7]) * 1000;
	printf("要素1SMBB：%f 要素2SMBB：%f\n", Feature[6].SMBB.area * 1000, Feature[7].SMBB.area * 1000);
	printf("相交面积1：%f 要素类型：%d %d\n", area1, Feature[6].FeatureType, Feature[7].FeatureType);
	printf("相交面积2：%f 要素类型：%d %d\n", area2, Feature[6].FeatureType, Feature[7].FeatureType);*/
	//PrintArea("FI_Area1.txt", FI_Area);
	//PrintArea("SI_Area1.txt", SI_Area);


}

//计算要素相交面积
double CalculateEnvelopeIntersection(Features<float > &Feature1, Features<float > &Feature2)
{
	double minx = MAX(Feature1.ENCenvelope.MinX, Feature2.ENCenvelope.MinX);
	double miny = MAX(Feature1.ENCenvelope.MinY, Feature2.ENCenvelope.MinY);
	double maxx = MIN(Feature1.ENCenvelope.MaxX, Feature2.ENCenvelope.MaxX);
	double maxy = MIN(Feature1.ENCenvelope.MaxY, Feature2.ENCenvelope.MaxY);
	if (minx > maxx || miny > maxy)
	{
		//printf("不相交\n");
		return 0;
	}
	else
	{
		//printf("相交面积：%f\n", (maxx - minx)*(maxy - miny));
		return (maxx - minx)*(maxy - miny);
	}
}
//计算单一要素四至范围面积
double CalculateEnvelopeOne(Features<float> &Feature)
{
	return (Feature.ENCenvelope.MaxX - Feature.ENCenvelope.MinX)*(Feature.ENCenvelope.MaxY - Feature.ENCenvelope.MinY);
}

//求最小外接矩形SMBB
void CalculateSMBB(Features<float > &Feature)
{
	OGRPoint poPoint;
	Feature.ENCGeometry->Centroid(&poPoint);
	Point<double> center;//几何体重心
	center.x = poPoint.getX();
	center.y = poPoint.getY();
	center.z = poPoint.getZ();
	SmallestMinBoundingBox<float> newSMBB;//旋转后的外接矩形
	
	newSMBB.area = CalculateEnvelopeOne(Feature);
	if (Feature.point.size() == 0) 
	{
		MessageBox(NULL, TEXT("该环为空"), TEXT("消息框"), MB_ICONINFORMATION | MB_YESNO);
	}
	else
	{
		double tempangle;
		for (double angle = 0.0f; angle <= 90; angle++)
		{
			SmallestMinBoundingBox<float> tempSMBB;//旋转后的外接矩形
			vector<Point<float>> vpt;//旋转后的点集
			for (int i = 0; i < Feature.point.size(); i++)
			{
				Point<float> tempPoint;
				tempPoint = RotatePonit(Feature.point[i], center, angle);
				vpt.push_back(tempPoint);
			}
			FindRectangle(vpt, tempSMBB);
			//printf("%0.3f \n", tempSMBB.area*1000);
			if (tempSMBB.area < newSMBB.area)
			{
				newSMBB = tempSMBB;
				tempangle = angle;
				//printf("%0.3f , %0.3f\n", tempSMBB.area * 1000, angle);
			}
		}
		for (int i = 0; i < 4; i++)
		{
			Point<float> tempPoint;
			tempPoint = RotatePonit(newSMBB.p[i], center, -tempangle);
			newSMBB.p[i] = tempPoint;
		}
	}
	Feature.SMBB = newSMBB;
/*	for (int i = 0; i < 4; i++)//坐标归一化
	{
		Feature.SMBB.p[i].x = (newSMBB.p[i].x - Center.x) / Center_ratio;
		Feature.SMBB.p[i].y = 0.0f;
		Feature.SMBB.p[i].z = -(newSMBB.p[i].z - Center.z) / Center_ratio;//z轴取负数，将数据坐标翻转，匹配视坐标系
	}
	Feature.SMBB.area = newSMBB.area;
	
	//printf("MaxX:%.3f ,MaxY:%.3f ,MinY:%.3f \n", Feature.SMBB.p[0].x, Feature.SMBB.p[0].z, Feature.SMBB.area);*/
}
// 某一点pt绕center旋转theta角度，zf,0706
Point<float> RotatePonit(Point<float> &pt, Point<double> &center, double theta)
{
	double x1 = pt.x;
	double z1 = pt.z;
	double x0 = center.x;
	double z0 = center.z;

	double Q = theta / 180 * 3.1415926;  //角度

	double x2, z2;
	x2 = (x1 - x0)*cos(Q) - (z1 - z0)*sin(Q) + x0;   //旋转公式
	z2 = (x1 - x0)*sin(Q) + (z1 - z0)*cos(Q) + z0;

	Point<float> rotatePoint;
	rotatePoint.x = x2;
	rotatePoint.y = 0.0f;
	rotatePoint.z = z2;
	return rotatePoint;
}
//求外接矩形
void FindRectangle(vector<Point<float>> &vpt,SmallestMinBoundingBox<float> &SMBB)
{
	if (vpt.size() == 0)
	{
		MessageBox(NULL, TEXT("该环为空"), TEXT("消息框"), MB_ICONINFORMATION | MB_YESNO);
	}
	else
	{
		double Xmax = 0;
		double Zmax = 0;
		double Xmin = 60000000;  //最小值不能初始为0，
		double Zmin = 1000000000;  //最小值不能初始为0，
		for (int i = 0; i<vpt.size(); i++)
		{
			double tempx = vpt[i].x;
			double tempz = vpt[i].z;
			if (tempx >= Xmax) Xmax = tempx;  //最大x，
			if (tempz >= Zmax) Zmax = tempz;  //最大z，
			if (tempx <= Xmin) Xmin = tempx;  //最小x
			if (tempz <= Zmin) Zmin = tempz;  //最小z
		}
		SMBB.p[0].x = Xmax; 
		SMBB.p[0].z = Zmax;
		SMBB.p[1].x = Xmax;
		SMBB.p[1].z = Zmin;
		SMBB.p[2].x = Xmin;
		SMBB.p[2].z = Zmin;
		SMBB.p[3].x = Xmin;
		SMBB.p[3].z = Zmax;
		SMBB.area = (Xmax - Xmin) * (Zmax - Zmin);
	}
}

//计算要素SMBB相交面积
double CalculateSMBBIntersection(Features<float > &Feature1, Features<float > &Feature2)
{
	//获取两个矩形的顶点
	vector<Point<float>> Feature1point, Feature2point, Allpoint;
	Point<float> CrossPoint;//相交点
	for (int i = 0; i < 4; i++)
	{
		Feature1point.push_back(Feature1.SMBB.p[i]);
		Feature2point.push_back(Feature2.SMBB.p[i]);

	}

	//判定各个顶点是否在另一个矩形框内
	for (int i = 0; i < 4; i++) {
		if (IsPointInRectangle(Feature1point[i], Feature2point))
			Allpoint.push_back(Feature1point[i]);
	}
	for (int i = 0; i < 4; i++) {
		if (IsPointInRectangle(Feature2point[i], Feature1point))
			Allpoint.push_back(Feature2point[i]);
	}

	//计算两个矩形各边的交点
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (segmentsCrossPoint(Feature1.SMBB.p[i], Feature1.SMBB.p[(i + 1) % 4], Feature2.SMBB.p[j], Feature2.SMBB.p[(j + 1) % 4], CrossPoint))
				Allpoint.push_back(CrossPoint);
		}
	}

	//将所有的点连起来组成一个凸多边形
	if (Allpoint.size() < 3) return 0;
	else {
		for (int i = 0; i < Allpoint.size(); i++) {
			for (int j = 0; j < Allpoint.size() - i - 1; j++) {
				if (Allpoint[j].x > Allpoint[j + 1].x)
					swap(Allpoint[j], Allpoint[j + 1]);
			}
		}
		vector<Point<float>> temp_shape_point(Allpoint);
		pair<float, float>line;
		line.first = ((Allpoint.end() - 1)->z - Allpoint.begin()->z) / ((Allpoint.end() - 1)->x - Allpoint.begin()->x);
		line.second = Allpoint.begin()->z - line.first*Allpoint.begin()->x;
		vector<int> ranking;
		int n = 1; int m = 0;
		for (int i = 0; i < int(Allpoint.size()); i++) {
			if (Allpoint[i].z <= line.first*Allpoint[i].x + line.second) {
				ranking.push_back(m);
				m++;
			}
			else {
				ranking.push_back(Allpoint.size() - n);
				n++;
			}
		}
		for (int i = 0; i < int(ranking.size()); i++) {
			Allpoint[ranking[i]] = temp_shape_point[i];
		}
	}


	//将凸多变形分成多个三角形来计算面积
	double sumS = 0.0;
	for (int i = 1; i < Allpoint.size()-1; i++)
		sumS += TriangleArea(Allpoint[0], Allpoint[i], Allpoint[i + 1]);// n-2个三角形的面积和
	//printf("%f\n", sumS);																	
	return sumS;
	/*
	float area = 0.0;
	for (int i = 0; i < int(Allpoint.size() - 1); i++) {
		area += Allpoint[i].x*Allpoint[i + 1].z - Allpoint[i + 1].x*Allpoint[i].z;
	}
	area = area / 2;
	int i = Allpoint.size() - 1;
	area = area - (Allpoint[0].x*Allpoint[i].z - Allpoint[i].x*Allpoint[0].z) / 2;

	//cout << "area:" << area << endl;
	return area;*/
}
// 计算 |p1 p2| X |p1 p|
float GetCross(Point<float>& p1, Point<float>& p2, Point<float>& p)
{
	return (p2.x - p1.x) * (p.z - p1.z) - (p.x - p1.x) * (p2.z - p1.z);
}
//判断一个点是否在另一矩形内
bool IsPointInRectangle(Point<float> &p, vector<Point<float>> all_shape_point)
{
	return (GetCross(all_shape_point[0], all_shape_point[1], p) * GetCross(all_shape_point[2], all_shape_point[3], p)) >= 0 
		&& (GetCross(all_shape_point[1], all_shape_point[2], p) * GetCross(all_shape_point[3], all_shape_point[0], p)) >= 0;
	//return false;  
}
//计算两个矩形各边的相交点
bool segmentsCrossPoint(Point<float> a, Point<float> b, Point<float> c, Point<float> d, Point<float>& CrossPoint) {
	float area_abc = (a.x - c.x) * (b.z - c.z) - (a.z - c.z) * (b.x - c.x);
	float area_abd = (a.x - d.x) * (b.z - d.z) - (a.z - d.z) * (b.x - d.x);
	if (area_abc*area_abd >= 0) {
		return false;
	}
	float area_cda = (c.x - a.x) * (d.z - a.z) - (c.z - a.z) * (d.x - a.x);
	float area_cdb = area_cda + area_abc - area_abd;
	if (area_cda * area_cdb >= 0) {
		return false;
	}
	float t = area_cda / (area_abd - area_abc);
	float dx = t*(b.x - a.x);
	float dz = t*(b.z - a.z);
	CrossPoint.x = a.x + dx;
	CrossPoint.z = a.z + dz;
	return true;
}

//计算两点之间的距离
double DistanceTwoPoints(Point<float> V1,Point<float> V2) 
{
	return sqrt(double((V1.x - V2.x)*(V1.x - V2.x) + (V1.y - V2.y)*(V1.y - V2.y) + (V1.z - V2.z)*(V1.z - V2.z)));
}
//计算三角形面积
double TriangleArea(Point<float> V1, Point<float> V2, Point<float> V3) 
{
	double a = DistanceTwoPoints(V1, V2);
	double b = DistanceTwoPoints(V1, V3);
	double c = DistanceTwoPoints(V3, V2);
	double p = (a + b + c) / 2;
	return  sqrt(p * (p - a) * (p - b) * (p - c));
}

//输出面积
void PrintArea(string FileName, vector<double> &area) 
{
	ofstream outfile;

	outfile.open(FileName, ostream::app);        /*以加入模式打开文件*/
	outfile << "----------------------------one frame-------------------------" << endl;
	for (size_t i = 0; i<area.size(); i++)
	{
		outfile << area[i] << endl;
	}
	outfile.close();                                 /*关闭文件*/
}



////////////////////////////////////////////////////////////////////////////////
//DrawFeatures绘制
////////////////////////////////////////////////////////////////////////////////
//绘制立方体
void cube(GLfloat size, GLfloat position)
{

	glBegin(GL_QUADS);
/*	glColor3f(1.0, 1.0, 0.0);
	glVertex3f(size, size, -size);
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(-size, size, -size);
	glColor3f(0.0, 1.0, 1.0);
	glVertex3f(-size, size, size);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(size, size, size);
*/
	glColor3f(1.0, 0.0, 1.0);
	glVertex3f(size, -size, size);
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(-size, -size, size);
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(-size, -size, -size);
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(size, -size, -size);

	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(size, 0.0, size);
	glColor3f(0.0, 1.0, 1.0);
	glVertex3f(-size, 0.0, size);
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(-size, -size, size);
	glColor3f(1.0, 0.0, 1.0);
	glVertex3f(size, -size, size);

	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(size, -size, -size);
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(-size, -size, -size);
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(-size, 0.0, -size);
	glColor3f(1.0, 1.0, 0.0);
	glVertex3f(size, 0.0, -size);

	glColor3f(0.0, 1.0, 1.0);
	glVertex3f(-size, 0.0, size);
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(-size, 0.0, -size);
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(-size, -size, -size);
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(-size, -size, size);

	glColor3f(1.0, 1.0, 0.0);
	glVertex3f(size, 0.0, -size);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(size, 0.0, size);
	glColor3f(1.0, 0.0, 1.0);
	glVertex3f(size, -size, size);
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(size, -size, -size);
	glEnd();
	
	//printf("1");
}

//小立方体
void small_cube(GLfloat x, GLfloat y, GLfloat z, GLfloat position_longitude, GLfloat position_latitudde)
{
	//GLfloat color = 1.0;
	//GLfloat alpha = 0.5;

	glBegin(GL_QUADS);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, y, z + position_latitudde);
	
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, -y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, -y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, -y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, -y, -z + position_latitudde);

	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, -y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, -y, z + position_latitudde);

	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, -y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, -y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, y, -z + position_latitudde);

	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, -y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(-x + 0.5*position_longitude, -y, z + position_latitudde);

	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, y, -z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, -y, z + position_latitudde);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(x + 0.5*position_longitude, -y, -z + position_latitudde);

	glEnd();

}

//浮标
void BouyCube(Point<float> BouyPosition, GLfloat position_longitude, GLfloat position_latitudde)
{
	
	//GLfloat color = 1.0;
	//GLfloat alpha = 0.5;
	unsigned int img = loadPNGTexture("E:\\program\\oceanFFT\\data\\BouyBMP\\左侧标.png");
	bouyTextures[0] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\data\\BouyBMP\\左侧标.bmp");
	bouyTextures[1] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\data\\BouyBMP\\DZ.bmp");
	/*bouyTextures[1] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\sky2\\front.bmp");
	glDisable(GL_DEPTH_TEST);*/

	
	glPolygonMode(GL_FRONT, GL_FILL);// 设置正面为填充模式
	glPolygonMode(GL_BACK, GL_LINE);// 设置反面为线形模式
	glFrontFace(GL_CCW);//切换正反面

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, img);

	glBegin(GL_QUADS);
	// Front Face
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(0.3*BouyPosition.x + position_longitude, BouyPosition.y, 0.3*BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-0.3*BouyPosition.x + position_longitude, BouyPosition.y, 0.3*BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	// Back Face
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-0.3*BouyPosition.x + position_longitude, BouyPosition.y, -0.3*BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(0.3*BouyPosition.x + position_longitude, BouyPosition.y, -0.3*BouyPosition.z + position_latitudde);
	// Left Face
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-0.3*BouyPosition.x + position_longitude, BouyPosition.y, 0.3*BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-0.3*BouyPosition.x + position_longitude, BouyPosition.y, -0.3*BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	// Right face
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(0.3*BouyPosition.x + position_longitude, BouyPosition.y, -0.3*BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(0.3*BouyPosition.x + position_longitude, BouyPosition.y, 0.3*BouyPosition.z + position_latitudde);
	glEnd();
/*
	GLfloat vertex[4];

	const GLfloat delta_angle = 2.0*M_PI / num_segments;
	glBegin(GL_TRIANGLE_FAN);

	vertex[0] = cx;
	vertex[1] = cy;
	vertex[2] = cz;
	vertex[3] = 1.0;
	glVertex4fv(vertex);

	//draw the vertex on the contour of the circle   
	for (int i = 0; i < num_segments; i++)
	{
		vertex[0] = std::cos(delta_angle*i) * r + cx;
		vertex[1] = std::sin(delta_angle*i) * r + cy;
		vertex[2] = cz;
		vertex[3] = 1.0;
		glVertex4fv(vertex);
	}

	vertex[0] = 1.0 * r + cx;
	vertex[1] = 0.0 * r + cy;
	vertex[2] = cz;
	vertex[3] = 1.0;
	glVertex4fv(vertex);
	glEnd();

	glBegin(GL_QUADS);
	//Top Face
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(-BouyPosition.x + position_longitude, BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(BouyPosition.x + position_longitude, BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(BouyPosition.x + position_longitude, BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(-BouyPosition.x + position_longitude, BouyPosition.y, -BouyPosition.z + position_latitudde);
	// Bottom Face
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f); 
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	// Front Face
	glTexCoord2f(0.0f, 0.0f); 
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f); 
	glVertex3f(BouyPosition.x + position_longitude, BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f); 
	glVertex3f(-BouyPosition.x + position_longitude, BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	// Back Face
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-BouyPosition.x + position_longitude, BouyPosition.y, -BouyPosition.z + position_latitudde);
 	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(BouyPosition.x + position_longitude, BouyPosition.y, -BouyPosition.z + position_latitudde);
    // Left Face
	glTexCoord2f(1.0f, 0.0f);	
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-BouyPosition.x + position_longitude, BouyPosition.y, BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-BouyPosition.x + position_longitude, BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	// Right face
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, BouyPosition.z + position_latitudde);	
	glTexCoord2f(1.0f, 0.0f); 
	glVertex3f(BouyPosition.x + position_longitude, -BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(BouyPosition.x + position_longitude, BouyPosition.y, -BouyPosition.z + position_latitudde);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(BouyPosition.x + position_longitude, BouyPosition.y, BouyPosition.z + position_latitudde);

	glEnd();
*/
}

//加载PNG图像
gl_texture_t * ReadPNGFromFile(const char *filename)
{
	gl_texture_t *texinfo;
	png_byte magic[8];
	png_structp png_ptr;
	png_infop info_ptr;
	int bit_depth, color_type;
	FILE *fp = NULL;
	png_bytep *row_pointers = NULL;
	png_uint_32 w, h;
	int i;
	/* Open image file */
	fopen_s(&fp, filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "error: couldn't open \"%s\"!\n", filename);
		return NULL;
	}
	/* Read magic number */
	fread(magic, 1, sizeof(magic), fp);
	/* Check for valid magic number */
	if (!png_check_sig(magic, sizeof(magic)))
	{
		fprintf(stderr, "error: \"%s\" is not a valid PNG image!\n", filename);
		fclose(fp);
		return NULL;
	}
	/* Create a png read struct */
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr)
	{
		fclose(fp);
		return NULL;
	}
	/* Create a png info struct */
	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
	{
		fclose(fp);
		png_destroy_read_struct(&png_ptr, NULL, NULL);
		return NULL;
	}
	/* Create our OpenGL texture object */
	texinfo = (gl_texture_t *)malloc(sizeof(gl_texture_t));
	/* Initialize the setjmp for returning properly after a libpng error occured */
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		fclose(fp);
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		if (row_pointers) free(row_pointers);
		if (texinfo) {
			if (texinfo->texels)
				free(texinfo->texels);
			free(texinfo);
		}
		return NULL;
	}
	/* Setup libpng for using standard C fread() function with our FILE pointer */
	png_init_io(png_ptr, fp);
	/* Tell libpng that we have already read the magic number */
	png_set_sig_bytes(png_ptr, sizeof(magic));
	/* Read png info */
	png_read_info(png_ptr, info_ptr);
	/* Get some usefull information from header */
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	/* Convert index color images to RGB images */
	if (color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png_ptr);
	/* Convert 1-2-4 bits grayscale images to 8 bits grayscale. */
	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png_ptr);

	if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png_ptr);
	if (bit_depth == 16) png_set_strip_16(png_ptr);
	else if (bit_depth < 8) png_set_packing(png_ptr);
	/* Update info structure to apply transformations */
	png_read_update_info(png_ptr, info_ptr);
	/* Retrieve updated information */
	png_get_IHDR(png_ptr, info_ptr, &w, &h, &bit_depth, &color_type, NULL, NULL, NULL);
	texinfo->width = w;
	texinfo->height = h;
	/* Get image format and components per pixel */
	GetPNGtextureInfo(color_type, texinfo);
	/* We can now allocate memory for storing pixel data */
	texinfo->texels = (GLubyte *)malloc(sizeof(GLubyte) * texinfo->width * texinfo->height * texinfo->internalFormat);
	/* Setup a pointer array. Each one points at the begening of a row. */
	row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * texinfo->height);
	for (i = 0; i < texinfo->height; ++i)
	{
		row_pointers[i] = (png_bytep)(texinfo->texels + ((texinfo->height - (i + 1)) * texinfo->width * texinfo->internalFormat));
	}
	/* Read pixel data using row pointers */
	png_read_image(png_ptr, row_pointers);
	/* Finish decompression and release memory */
	png_read_end(png_ptr, NULL);
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	/* We don't need row pointers anymore */
	free(row_pointers);
	fclose(fp);
	return texinfo;
}
GLuint loadPNGTexture(const char *filename)
{
	gl_texture_t *png_tex = NULL;
	GLuint tex_id = 0;
	GLint alignment;
	png_tex = ReadPNGFromFile(filename);
	if (png_tex && png_tex->texels)
	{
		/* Generate texture */
		glGenTextures(1, &png_tex->id);
		glBindTexture(GL_TEXTURE_2D, png_tex->id);
		/* Setup some parameters for texture filters and mipmapping */
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		glGetIntegerv(GL_UNPACK_ALIGNMENT, &alignment);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, png_tex->internalFormat, png_tex->width, png_tex->height, 0, png_tex->format, GL_UNSIGNED_BYTE, png_tex->texels);
		glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
		tex_id = png_tex->id;
		/* OpenGL has its own copy of texture data */
		free(png_tex->texels);
		free(png_tex);
	}
	return tex_id;
}
void GetPNGtextureInfo(int color_type, gl_texture_t *texinfo)
{
	switch (color_type)
	{
	case PNG_COLOR_TYPE_GRAY:
		texinfo->format = GL_LUMINANCE;
		texinfo->internalFormat = 1;
		break;

	case PNG_COLOR_TYPE_GRAY_ALPHA:
		texinfo->format = GL_LUMINANCE_ALPHA;
		texinfo->internalFormat = 2;
		break;

	case PNG_COLOR_TYPE_RGB:
		texinfo->format = GL_RGB;
		texinfo->internalFormat = 3;
		break;

	case PNG_COLOR_TYPE_RGB_ALPHA:
		texinfo->format = GL_RGBA;
		texinfo->internalFormat = 4;
		break;

	default:
		/* Badness */
		break;
	}
}

//坐标轴
void Coordinate()
{
	glBegin(GL_LINES);
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(-2.0, 0.0, 0.0);
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(2.0, 0.0, 0.0);
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(0.0, -2.0, 0.0);
	glColor3f(1.0, 1.0, 0.0);
	glVertex3f(0.0, 2.0, 0.0);
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(0.0, 0.0, -2.0);
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(0.0, 0.0, 2.0);
	glEnd();
}

//绘制圆
void getPointOfCircle(float radius, Point<float> center) {
	int N = 100;
	for (int i = 0; i<N; i++)
		circle.push_back(Point<float>(radius*cos(2 * pi / N*i) + center.x, center.y, radius*sin(2 * pi / N*i) + center.z));
	//circle.push_back(Point<float>(center.x, center.y, center.z));
}
void drawCircle(float start_angle, float end_angle)
{
	
	glBegin(GL_LINE_LOOP);
	for (int i = 0; i<circle.size(); i++) {
		glVertex3f(circle[i].x, circle[i].y, circle[i].z);
		//i += 4;
	}
	glEnd();
	circle.clear();
/*
	//printf("%d\n", circle.size());
	if (start_angle > 360)
	{
		start_angle -= 360;
	}
	int start_N = floor(start_angle / 360 * (circle.size()-1)) + 1;
	int end_N = floor(end_angle / 360 * (circle.size()-1)) + 1;
    printf("%d\n", start_N);
	printf("%d\n", end_N);
	printf("%d\n", circle.size());
	glBegin(GL_POLYGON);
	for (int i = start_N; i<end_N; i++) {
		//glColor4f(1.0, 0.0, 1.0, 0.4);
        //glVertex3f(circle[i-1].x, circle[i-1].y, circle[i-1].z);
		glColor4f(1.0, 0.0, 1.0, 0.4);
		glVertex3f(circle[i].x, circle[i].y, circle[i].z);
		
		//i += 4;
	}
    glColor4f(1.0, 0.0, 1.0, 0.4);
	glVertex3f(circle[circle.size()-1].x, circle[circle.size()-1].y, circle[circle.size()-1].z);
	printf("%f\n", circle[circle.size()-1].x);
	glEnd();
	*/
}

//绘制圆弧、半圆
void getPointOfArc(float radius, float start_angle, float end_angle, Point<float> center) {
	int N = 100;
	if (start_angle < end_angle) {
		float diff = end_angle - start_angle;
		for (int i = 0; i < N; i++)
			arc.push_back(Point<float>(radius*cos((start_angle + diff / N*i) / 360.0 * 2 * pi) + center.x, center.y, radius*sin((start_angle + diff / N*i) / 360.0 * 2 * pi) + center.z));
		arc.push_back(Point<float>(radius*cos(end_angle / 360.0 * 2 * pi) + center.x, center.y, radius*sin(end_angle / 360.0 * 2 * pi) + center.z));//将圆弧终点加上
		arc.push_back(Point<float>(center.x, center.y, center.z));//将圆点加上
	}
	else {
		float diff = end_angle - start_angle + 360.0f;
		for (int i = 0; i < N; i++)
			arc.push_back(Point<float>(radius*cos((start_angle + diff / N*i) / 360.0 * 2 * pi) + center.x, center.y, radius*sin((start_angle + diff / N*i) / 360.0 * 2 * pi) + center.z));
		arc.push_back(Point<float>(radius*cos(end_angle / 360.0 * 2 * pi) + center.x, center.y, radius*sin(end_angle / 360.0 * 2 * pi) + center.z));//将圆弧终点加上
		arc.push_back(Point<float>(center.x, center.y, center.z));//将圆点加上
	}
}
void drawArc() {
	//glBegin(GL_LINE_STRIP);
	glBegin(GL_POLYGON);
	//glColor4f(1.0, 0.0, 0.0, 0.5);
	for (int i = 0; i<arc.size(); i++)
	{
		glVertex3f(arc[i].x, arc[i].y, arc[i].z);
	}
	
	glVertex3f(arc[arc.size() - 1].x, arc[arc.size() - 1].y, arc[arc.size() - 1].z);
	glEnd();
	arc.clear();
}

//绘制点
void DrawPoints(vector<Point<float>> &DrawPoint_P, float size, float4 color)
{
	glPointSize(size);
	glBegin(GL_POINTS);
	for (int i = 0; i < DrawPoint_P.size(); i++)//画点
	{
		glColor4f(color.x, color.y, color.z, color.w);
		glVertex3f(DrawPoint_P[i].x, DrawPoint_P[i].y, DrawPoint_P[i].z);;
	}
	glEnd();
	glFlush();
}
//绘制线
void DrawLine(vector<Point<float>> &DrawPoint_L, bool point, float4 color)
{
	if (point == true)
	{
		glPointSize(3.0f);
		glBegin(GL_POINTS);
		for (int i = 0; i < DrawPoint_L.size(); i++)
		{
			glColor3f(1.0, 1.0, 1.0);
			glVertex3f(DrawPoint_L[i].x, 0.0, DrawPoint_L[i].z);
		}
		glEnd();
	}

	glBegin(GL_QUADS);
	for (int i = 1; i < DrawPoint_L.size(); i++)
	{
		if (DrawPoint_L[i].y == DrawPoint_L[i - 1].y)
		{
			glColor4f(color.x, color.y, color.z, color.w);
			glVertex3f(DrawPoint_L[i - 1].x, 0.01, DrawPoint_L[i - 1].z);
			glColor4f(color.x, color.y, color.z, color.w);
			glVertex3f(DrawPoint_L[i].x, 0.01, DrawPoint_L[i].z);
			glColor4f(color.x, color.y, color.z, color.w);
			glVertex3f(DrawPoint_L[i].x, -0.01, DrawPoint_L[i].z);
			glColor4f(color.x, color.y, color.z, color.w);
			glVertex3f(DrawPoint_L[i - 1].x, -0.01, DrawPoint_L[i - 1].z);
		}
	}
	glEnd();
}
//绘制面
void DrawAre(vector<Point<float>> &DrawPoint_A, float Point_y, bool point, bool top, float4 color)
{
	int AreNum = 1;
	if (point == true)
	{
		glPointSize(5.0f);
		glBegin(GL_POINTS);
		for (int i = 0; i < DrawPoint_A.size(); i++)
		{
			glColor3f(1.0, i / DrawPoint_A.size(), 1.0);
			glVertex3f(DrawPoint_A[i].x, Point_y, DrawPoint_A[i].z);
		}
		glEnd();
	}
	glBegin(GL_QUADS);
	for (int i = 1; i < DrawPoint_A.size(); i++)
	{
		if (DrawPoint_A[i].y == DrawPoint_A[i - 1].y)
		{
			glColor4f(color.x, color.y, color.z, color.w);
			glVertex3f(DrawPoint_A[i - 1].x, 0.01 + Point_y, DrawPoint_A[i - 1].z);
			glColor4f(color.x, color.y, color.z, color.w);
			glVertex3f(DrawPoint_A[i].x, 0.01 + Point_y, DrawPoint_A[i].z);
			glColor4f(color.x, color.y, color.z, color.w);
			glVertex3f(DrawPoint_A[i].x, -0.01 + Point_y, DrawPoint_A[i].z);
			glColor4f(color.x, color.y, color.z, color.w);
			glVertex3f(DrawPoint_A[i - 1].x, -0.01 + Point_y, DrawPoint_A[i - 1].z);
		}
		else
		{
			AreNum++;
		}
	}
	glEnd();
	if (top == true)
	{
		for (int i = 0; i < DrawPoint_A.size(); i++)
		{/*
			MinBoundingBox<float> MBB;
			if (i == 0 || DrawPoint_A[i].y != DrawPoint_A[i - 1].y)
			{
				MBB.left_up.z = MBB.right_down.z = DrawPoint_A[i].z;
				MBB.left_up.x = MBB.right_down.x = DrawPoint_A[i].x;
			}*/
			if (DrawPoint_A[i].y == DrawPoint_A[i - 1].y && i % 100 != 0)
			{/*
				if (DrawPoint_A[i].z > MBB.left_up.z)
				{
					MBB.left_up.z = DrawPoint_A[i].z;
				}
				if (DrawPoint_A[i].z < MBB.right_down.z)
				{
					MBB.right_down.z = DrawPoint_A[i].z;
				}
				if (DrawPoint_A[i].x > MBB.left_up.x)
				{
					MBB.left_up.x = DrawPoint_A[i].x;
				}
				if (DrawPoint_A[i].x < MBB.right_down.x)
				{
					MBB.right_down.x = DrawPoint_A[i].x;
				}*/
				glBegin(GL_POLYGON);
				glColor4f(1.0f, 0.0f, 1.0f, 0.5);
				glVertex3f(DrawPoint_A[i - 1].x, 0.01, DrawPoint_A[i - 1].z);
				glColor4f(1.0f, 0.0f, 1.0f, 0.5);
				glVertex3f(DrawPoint_A[i].x, 0.01, DrawPoint_A[i].z);
			}
			if (DrawPoint_A[i].y != DrawPoint_A[i + 1].y)
			{
				glEnd();
				//MinimumBoundingBox(MBB);
				/*printf("%f\n", left_up.x);
				printf("%f\n", right_down.x);
				printf("%s\n", "No data");*/
			}	
		}
	}

}

/*
void PolyScan()
{

	int MaxY = 0;
	int i;
	for (i = 0; i<POINTNUM; i++)
		if (polypoint[i].y>MaxY)
			MaxY = polypoint[i].y;


	AET *pAET = new AET;
	pAET->next = NULL;


	NET *pNET[1024];
	for (i = 0; i <= MaxY; i++)
	{
		pNET[i] = new NET;
		pNET[i]->next = NULL;
	}
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.0, 0.0, 0.0);
	glBegin(GL_POINTS);

	for (i = 0; i <= MaxY; i++)
	{
		for (int j = 0; j<POINTNUM; j++)
			if (polypoint[j].y == i)
			{
				if (polypoint[(j - 1 + POINTNUM) % POINTNUM].y>polypoint[j].y)
				{
					NET *p = new NET;
					p->x = polypoint[j].x;
					p->ymax = polypoint[(j - 1 + POINTNUM) % POINTNUM].y;
					p->dx = (polypoint[(j - 1 + POINTNUM) % POINTNUM].x - polypoint[j].x) / (polypoint[(j - 1 + POINTNUM) % POINTNUM].y - polypoint[j].y);
					p->next = pNET[i]->next;
					pNET[i]->next = p;

				}
				if (polypoint[(j + 1 + POINTNUM) % POINTNUM].y>polypoint[j].y)
				{
					NET *p = new NET;
					p->x = polypoint[j].x;
					p->ymax = polypoint[(j + 1 + POINTNUM) % POINTNUM].y;
					p->dx = (polypoint[(j + 1 + POINTNUM) % POINTNUM].x - polypoint[j].x) / (polypoint[(j + 1 + POINTNUM) % POINTNUM].y - polypoint[j].y);
					p->next = pNET[i]->next;
					pNET[i]->next = p;
				}
			}
	}

	for (i = 0; i <= MaxY; i++)
	{

		NET *p = pAET->next;
		while (p)
		{
			p->x = p->x + p->dx;
			p = p->next;
		}

		AET *tq = pAET;
		p = pAET->next;
		tq->next = NULL;
		while (p)
		{
			while (tq->next && p->x >= tq->next->x)
				tq = tq->next;
			NET *s = p->next;
			p->next = tq->next;
			tq->next = p;
			p = s;
			tq = pAET;
		}
		AET *q = pAET;
		p = q->next;
		while (p)
		{
			if (p->ymax == i)
			{
				q->next = p->next;
				delete p;
				p = q->next;
			}
			else
			{
				q = q->next;
				p = q->next;
			}
		}
		p = pNET[i]->next;
		q = pAET;
		while (p)
		{
			while (q->next && p->x >= q->next->x)
				q = q->next;
			NET *s = p->next;
			p->next = q->next;
			q->next = p;
			p = s;
			q = pAET;
		}

		p = pAET->next;
		while (p && p->next)
		{
			for (float j = p->x; j <= p->next->x; j++)
				glVertex2i(static_cast<int>(j), i);
			p = p->next->next;
		}


	}
	glEnd();
	glFlush();
}
*/
/*
//绘制最小包围盒MBB
void MinimumBoundingBox(MinBoundingBox<float> &MBB)
{
	glBegin(GL_LINE_LOOP);
	glColor3f(1.0, 1.0, 1.0);//上
	glVertex3f(MBB.right_down.x, 0.01, MBB.right_down.z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(MBB.left_up.x, 0.01, MBB.right_down.z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(MBB.left_up.x, 0.01, MBB.left_up.z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(MBB.right_down.x, 0.01, MBB.left_up.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glColor3f(1.0, 1.0, 1.0);//下
	glVertex3f(MBB.right_down.x, -0.01, MBB.left_up.z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(MBB.left_up.x, -0.01, MBB.left_up.z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(MBB.left_up.x, -0.01, MBB.right_down.z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(MBB.right_down.x, -0.01, MBB.right_down.z);
	glEnd();
	glBegin(GL_LINES);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(MBB.left_up.x, -0.01, MBB.left_up.z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(MBB.left_up.x, 0.01, MBB.left_up.z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(MBB.left_up.x, -0.01, MBB.right_down.z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(MBB.left_up.x, 0.01, MBB.right_down.z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(MBB.right_down.x, -0.01, MBB.left_up.z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(MBB.right_down.x, 0.01, MBB.left_up.z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(MBB.right_down.x, -0.01, MBB.right_down.z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(MBB.right_down.x, 0.01, MBB.right_down.z);
	glEnd();

}*/
//绘制最小包围盒SMBB
void DrewSMBB(SmallestMinBoundingBox<float> &SMBB)
{
	glBegin(GL_LINE_LOOP);
	glColor3f(1.0, 1.0, 1.0);//上
	glVertex3f(SMBB.p[0].x, 0.01, SMBB.p[0].z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(SMBB.p[1].x, 0.01, SMBB.p[1].z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(SMBB.p[2].x, 0.01, SMBB.p[2].z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(SMBB.p[3].x, 0.01, SMBB.p[3].z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glColor3f(1.0, 1.0, 1.0);//下
	glVertex3f(SMBB.p[0].x, -0.01, SMBB.p[0].z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(SMBB.p[1].x, -0.01, SMBB.p[1].z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(SMBB.p[2].x, -0.01, SMBB.p[2].z);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(SMBB.p[3].x, -0.01, SMBB.p[3].z);
	glEnd();
	glBegin(GL_LINES);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(SMBB.p[0].x, -0.01, SMBB.p[0].z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(SMBB.p[0].x, 0.01, SMBB.p[0].z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(SMBB.p[1].x, -0.01, SMBB.p[1].z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(SMBB.p[1].x, 0.01, SMBB.p[1].z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(SMBB.p[2].x, -0.01, SMBB.p[2].z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(SMBB.p[2].x, 0.01, SMBB.p[2].z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(SMBB.p[3].x, -0.01, SMBB.p[3].z);
	glColor3f(1.0, 1.0, 1.0);//侧
	glVertex3f(SMBB.p[3].x, 0.01, SMBB.p[3].z);
	glEnd();
}
//绘制要素四至范围
void DrewEnvelope(OGREnvelope &ENCenvelope, float4 color)
{
	//printf("MaxX:%.3f ,MaxY:%.3f ,MinX:%.3f ,MinY:%.3f \n", ENCenvelope.MaxX, ENCenvelope.MaxY, ENCenvelope.MinX, ENCenvelope.MinY);
	glBegin(GL_LINE_LOOP);
	glColor4f(color.x, color.y, color.z, color.w);//上
	glVertex3f(ENCenvelope.MaxX, 0.01, ENCenvelope.MaxY);
	glColor4f(color.x, color.y, color.z, color.w);
	glVertex3f(ENCenvelope.MaxX, 0.01, ENCenvelope.MinY);
	glColor4f(color.x, color.y, color.z, color.w);
	glVertex3f(ENCenvelope.MinX, 0.01, ENCenvelope.MinY);
	glColor4f(color.x, color.y, color.z, color.w);
	glVertex3f(ENCenvelope.MinX, 0.01, ENCenvelope.MaxY);
	glEnd();
}
/*
void dd()
{
	GLfloat curColor[4];
	glGetFloatv(GL_CURRENT_COLOR, curColor);

	glColor3f(1.0f, 1.0f, 1.0f);
	int vertice_number = m_struTriangleInfo.m_iVertexNumber;
	glm::vec3*  vertices = m_struTriangleInfo.m_vVertice;
	glm::vec2*  texcoords = m_struTriangleInfo.m_vTexcoord;
	// enable and specify pointers to vertex arrays


	if (vertices)
	{
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, vertices);
	}

	if (texcoords)
	{
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(2, GL_FLOAT, 0, texcoords);
	}

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, TextureID);//TextureID是之前绑定的纹理对象
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (float*)(&(vec4(1.0))));
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (float*)(&(vec4(0.1, 0.1, 0.1, 1.0))));

	glDrawElements(GL_TRIANGLES, vertice_number, GL_UNSIGNED_INT, m_struTriangleInfo.m_vTriangle);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);


	if (vertices)
	{
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	if (texcoords)
	{
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	}
	glColor3f(curColor[0], curColor[1], curColor[2]);
	
}*/

//绘制要素
void DrawFeatures() 
{
	std::string filename = "E:\\program\\oceanFFT\\light.obj";

	TriangleMesh mesh;

	//loadObj(filename, mesh);
	
/*	struct Point centerofcircle = { 0.1f,0.1f,0.1f };
	getPointOfCircle(0.1f, centerofcircle);
	drawCircle();
	//getPointOfArc(0.2f, 30.0f, 150.0f);
	//drawArc();
	//DrawLine(TSSBND_L, false);
	DrawAre(RESARE_A, false, true);
		glPointSize(10.0f);
		glBegin(GL_POINTS);
		glColor3f(1.0, 1.0, 1.0);
		for (int i = 0; i < 10; i++)
		{
			glVertex3f(LNDARE_A[1095+i].x, 0.0, LNDARE_A[1095+i].z);
		}
		
		glEnd();*/

	static GLfloat angle_start = 0.0;
	static GLfloat angle_end = 20.0;
	angle_start += 2.0;
	angle_end += 2.0;

	float f[4] = { 1.0,1.0,1.0,1.0 };
	glEnable(GL_FOG);//打开雾效果
	glFogfv(GL_FOG_COLOR, f);//设置雾颜色，f是一个指定颜色的数组float f[4]
	glFogf(GL_FOG_START, 0.0f);//设置雾从多远开始
	glFogf(GL_FOG_END, 100.0f);//设置雾从多远结束
	glFogi(GL_FOG_MODE, GL_LINEAR);//设置使用哪种雾，共有三中雾化模式

	Point<float> centerofcircle;
	centerofcircle = { 0.1f,0.1f,0.1f };
	Point<float> centerofcircle2;
	centerofcircle2 = { 0.1f,0.1f,0.2f };
	//drawCircle(angle_start++, angle_end++);
	//drawArc();

	//设置透明
	glEnable(GL_BLEND); // 启用混合
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//设置混合因子
	glDepthMask(GL_FALSE);// 下面将绘制半透明物体了，因此将深度缓冲设置为只读
	glPushMatrix();
/*
	for (int i = 0; i < SOUNDG_P.size(); i++)
	{
		Point<float> center = { SOUNDG_P[i].x, 0.1f, SOUNDG_P[i].z };
		getPointOfCircle(0.01f, center);
		getPointOfArc(0.01f, angle_start, angle_end, center);
		glColor4f(0.0f, 1.0f, 0.0f, 0.5);
		drawCircle(1, 1);
		glColor4f(0.0, 1.0, 0.0, 0.5);
		drawArc();
	}

	getPointOfCircle(0.01f, centerofcircle2);
	getPointOfArc(0.01f, angle_start, angle_end, centerofcircle2);
	glColor4f(0.0f, 1.0f, 0.0f, 0.5);
	drawCircle(1,1);
	drawArc();*/
	getPointOfCircle(0.1f, centerofcircle);
	getPointOfArc(0.1f, angle_start, angle_end, centerofcircle);
	glColor4f(1.0f, 0.0f, 0.0f, 0.5);
	//drawCircle(1, 1);
	//drawArc();

	//绘制要素
	float y = 0.002f;
	
/*	for (int i = 0; i < ENCFeatureIndex.size(); i++)
	{
		if (wkbFlatten(ENCFeatureIndex[i].ENCGeometry->getGeometryType()) == wkbPoint && drawseaPoints)
		{
			DrawPoints(ENCFeatureIndex[i].point, 3.0f, whitecolor);
		}
		if (wkbFlatten(ENCFeatureIndex[i].ENCGeometry->getGeometryType()) == wkbLineString && drawseaLine)
		{
			DrawLine(ENCFeatureIndex[i].point, false, magentacolor);
		}
		if (wkbFlatten(ENCFeatureIndex[i].ENCGeometry->getGeometryType()) == wkbPolygon && drawseaAre)
		{
			DrawAre(ENCFeatureIndex[i].point, y, false, false, sandybrowncolor);
		}
		if (wkbFlatten(ENCFeatureIndex[i].ENCGeometry->getGeometryType()) == wkbMultiPoint && drawseaPoints)
		{
			DrawPoints(ENCFeatureIndex[i].point, 3.0f, whitecolor);
		}
	}*/

	DrewEnvelope(Searchenvelope, yellowcolor);
	if (drawseaLine)
	{
		for (int i = 0; i < QRLnodes[0].size(); i++)
		{
			DrewEnvelope(QRLnodes[0][i]->ENCenvelope, whitecolor);
		}
	}
	if (drawseaAre)
	{
		for (int i = 0; i < Qnodes[0].size(); i++)
		{
			DrewEnvelope(Qnodes[0][i]->ENCenvelope, whitecolor);
		}
	}
/*	for (int i = 0; i < Rnodes[0].size(); i++)
	{
		DrewEnvelope(Rnodes[0][i]->envelope, greencolor);
	}
	if (drawseaPoints)
	{
		for (int i = 0; i < QRnodes[0].size(); i++)
		{
			DrewEnvelope(QRnodes[0][i]->envelope, whitecolor);
		}
	}*//*
	for (int i = 0; i < Nresult.size(); i++)
	{
		if (wkbFlatten(Nresult[i]->ENCGeometry->getGeometryType()) == wkbPoint && drawseaPoints)
		{
			glColor4f(0.5f, 0.5f, 0.0f, 0.5);
			DrawPoints(Nresult[i]->point, 3.0f);
		}
		if (wkbFlatten(Nresult[i]->ENCGeometry->getGeometryType()) == wkbLineString && drawseaLine)
		{
			//glColor4f(1.0f, 1.0f, 0.5f, 0.5);
			DrawLine(Nresult[i]->point, false);
		}
		if (wkbFlatten(Nresult[i]->ENCGeometry->getGeometryType()) == wkbPolygon && drawseaAre)
		{
			//glColor4f(1.0f, 0.0f, 0.0f, 0.5);
			DrawAre(Nresult[i]->point, y, false, false);
			//y += 0.002f;
		}
		if (wkbFlatten(Nresult[i]->ENCGeometry->getGeometryType()) == wkbMultiPoint && drawseaPoints)
		{
			glColor4f(1.0f, 1.0f, 1.0f, 0.5);
			DrawPoints(Nresult[i]->point, 3.0f);
		}
	}*/
	
	for (int i = 0; i < Rresult[0].size(); i++)
	{
		if (wkbFlatten(Rresult[0][i]->ENCGeometry->getGeometryType()) == wkbPoint && drawseaPoints)
		{
			DrawPoints(Rresult[0][i]->point, 3.0f, whitecolor);
		}
		if (wkbFlatten(Rresult[0][i]->ENCGeometry->getGeometryType()) == wkbLineString && drawseaPoints)
		{
			//glColor4f(1.0f, 0.0f, 0.0f, 0.5);
			DrawLine(Rresult[0][i]->point, false, magentacolor);
		}
		if (wkbFlatten(Rresult[0][i]->ENCGeometry->getGeometryType()) == wkbPolygon && drawseaPoints)
		{
			//glColor4f(1.0f, 0.0f, 0.0f, 0.5);
			DrawAre(Rresult[0][i]->point, y, false, false, sandybrowncolor);
			//y += 0.002f;
		}
		if (wkbFlatten(Rresult[0][i]->ENCGeometry->getGeometryType()) == wkbMultiPoint && drawseaPoints)
		{
			DrawPoints(Rresult[0][i]->point, 3.0f, whitecolor);
		}
	}
	for (int i = 0; i < QRresult[0].size(); i++)
	{
		if (wkbFlatten(QRresult[0][i]->ENCGeometry->getGeometryType()) == wkbPoint && drawseaLine)
		{
			DrawPoints(QRresult[0][i]->point, 3.0f, whitecolor);
		}
		if (wkbFlatten(QRresult[0][i]->ENCGeometry->getGeometryType()) == wkbLineString && drawseaLine)
		{
			//glColor4f(1.0f, 0.0f, 0.0f, 0.5);
			DrawLine(QRresult[0][i]->point, false, magentacolor);
		}
		if (wkbFlatten(QRresult[0][i]->ENCGeometry->getGeometryType()) == wkbPolygon && drawseaLine)
		{
			//glColor4f(1.0f, 0.0f, 0.0f, 0.5);
			DrawAre(QRresult[0][i]->point, y, false, false, sandybrowncolor);
			//y += 0.002f;
		}
		if (wkbFlatten(QRresult[0][i]->ENCGeometry->getGeometryType()) == wkbMultiPoint && drawseaLine)
		{
			DrawPoints(QRresult[0][i]->point, 3.0f, whitecolor);
		}
	}
/*	for (int i = 6; i < 8; i++)
	{
		if (wkbFlatten(ENCFeatureIndex[i].ENCGeometry->getGeometryType()) == wkbPolygon)
		{
			DrewEnvelope(ENCFeatureIndex[i].ENCenvelope);
			DrewSMBB(ENCFeatureIndex[i].SMBB);
			DrawAre(ENCFeatureIndex[i].point, y, false, false);
		}
	}
*/

/*	for (int i = 0; i < FeatureIndex.size(); i++)
	{
		float y = 0.02f * i;
		if (FeatureIndex[i].shapetype == 1 && drawseaPoints)
		{
			//glColor4f(1.0f, 0.0f, 0.0f, 0.5);
			//DrawPoints(FeatureIndex[i].point);
		}
		if (FeatureIndex[i].shapetype == 3 && drawseaLine)
		{
			DrawLine(FeatureIndex[i].point, true);
		}
		if (FeatureIndex[i].shapetype == 5 && drawseaAre)
		{
			DrawAre(FeatureIndex[i].point, y, false, false);
		}
	}
	//绘制要素
	for (int i = 0; i < ENCFeatureIndex.size(); i++)
	{
		OGRFeature *EncFeature= ENCFeatureIndex[i];
		OGRGeometry *EncGeometry= EncFeature->GetGeometryRef();//获取要素中几何体的指针
		float y = 0.02f * i;			
		if (EncGeometry != NULL && wkbFlatten(EncGeometry->getGeometryType()) == wkbPoint && drawseaPoints)
		{
			vector<Point<float>> Point_P;
			OGRPoint *poPoint = (OGRPoint *)EncGeometry;
			Point<float> *P = (poPoint->getX(), poPoint->getY(), 0);
			normalization(*P);
			Point_P.push_back(*P);
			DrawPoints(Point_P);
			printf("%.3f,%.3f\n", poPoint->getX(), poPoint->getY());
			printf("%.3f,%.3f\n", P->x, P->y);
		}
	}

	
	if (drawseaPoints)
	{		
		DrawPoints(FeatureIndex[i].point);           //绘制水深点
		//printf("%d\n", LNDARE_A.size());
	}
	if (drawseaLine)
	{
		DrawLine(TSSBND_L, false);      //绘制分道通航方案边界
	}
	if (drawseaare)
	{
		DrawAre(LNDARE_A, false, true);        //绘制限制区
		//DrawAre(RESARE_A, false, true);       //绘制限制区
	}
*/
	glPopMatrix();
	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);
	
	/*
	glEnable(GL_LINE_STIPPLE);
    
	glLineWidth(2.0f);
	glColor3f(1.0f, 1.0f, 0.0f);
	glLineStipple(2, 0x0F0F);
	glBegin(GL_LINES);
	static GLfloat start = 0.0;

	glVertex3f(0.0f, 0.1f, 0.1f);
	glVertex3f(1.0f, 0.1f, 0.1f);
	glEnd();
	glDisable(GL_LINE_STIPPLE);
*/
	GLUquadric *pObj;
	pObj = gluNewQuadric();

	gluDeleteQuadric(pObj);
}
//创建菜单
void CreatMenu() 
{
	int nModeMenu;
	int nMainMenu;
	int nColorMenu;
	int nFeatureMenu;

/*	nModeMenu = glutCreateMenu(ProcessMenu);
	glutAddMenuEntry("正面多边形填充模式", 1);
	glutAddMenuEntry("正面线宽模型", 2);
	glutAddMenuEntry("正面点模式", 3);
	glutAddMenuEntry("反面多边形填充模式", 4);
	glutAddMenuEntry("反面多变形线宽模型", 5);
	glutAddMenuEntry("反面多变形点模式", 6);

	//再增加一个子菜单
	nColorMenu = glutCreateMenu(ProcessMenu);
	glutAddMenuEntry("平面明暗模式", 7);
	glutAddMenuEntry("光滑明暗模式", 8);*/

	//增加绘制要素菜单
	nFeatureMenu = glutCreateMenu(ProcessMenu);
	glutAddMenuEntry("DrawPoint", 1);
	glutAddMenuEntry("DrawLine", 2);
	glutAddMenuEntry("DrawAre", 3);

	//创建主菜单
	nMainMenu = glutCreateMenu(ProcessMenu);
	glutAddSubMenu("Feature", nFeatureMenu);
/*	glutAddSubMenu("多边形模式", nModeMenu);
	glutAddSubMenu("颜色模式", nColorMenu);
	glutAddMenuEntry("改变绕法", 9);*/

	glutAttachMenu(GLUT_RIGHT_BUTTON);//GLUT_LEFT_BUTTON   GLUT_MIDDLE_BUTTON   GLUT_RIGHT_BUTTON

}

void ProcessMenu(int value)
{
	switch (value)
	{
	case 1:
		drawseaPoints = !drawseaPoints;
		break;
	case 2:
		drawseaLine = !drawseaLine;
		break;
	case 3:
		drawseaAre = !drawseaAre;
		break;/*
	case 1:
		//修改多边形正面为填充模式
		glPolygonMode(GL_FRONT, GL_FILL);
		break;
	case 2:
		//修改多边形正面为线模式
		glPolygonMode(GL_FRONT, GL_LINE);
		break;
	case 3:
		//修改多变行正面为点填充模式
		glPolygonMode(GL_FRONT, GL_POINT);
		break;
	case 4:
		//修改多变形背面为填充模式
		glPolygonMode(GL_BACK, GL_FILL);
		break;
	case 5:
		//修改多边形背面面为线模式
		glPolygonMode(GL_BACK, GL_LINE);
		break;
	case 6:
		//修改多变行背面为点填充模式
		glPolygonMode(GL_BACK, GL_POINT);
		break;
	case 7:
		//设置多边形的阴影模式为平面明暗模式
		glShadeModel(GL_FLAT);
		break;
	case 8:
		//设置多变形的阴影模式为光滑明暗模式
		glShadeModel(GL_SMOOTH);
		break;
	case 9:
		//bwinding = !bwinding;
		break;
	case 10:
		drawseaPoints = !drawseaPoints;
		break;
	case 11:
		drawseaLine = !drawseaLine;
		break;
	case 12:
		drawseaAre = !drawseaAre;
		break;*/
	default: break;
	}
	//强制刷新
	glutPostRedisplay();
}

//绘制天空盒
void SkyBoxInit()
{
	//char *file = "E:\\program\\oceanFFT\\common\\image\\sky2";
	mTextures[0] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\sky2\\front.bmp");
	mTextures[1] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\sky2\\back.bmp");
	mTextures[2] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\sky2\\left.bmp");
	mTextures[3] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\sky2\\right.bmp");
	mTextures[4] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\sky2\\top.bmp");
	mTextures[5] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\sky2\\bottom.bmp");
	/*
	mTextures[0] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\2FRONT.BMP");
	mTextures[1] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\2BACK.BMP");
	mTextures[2] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\2LEFT.BMP");
	mTextures[3] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\2RIGHT.BMP");
	mTextures[4] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\2TOP.BMP");
	mTextures[5] = CreateTexture2DFromBMP("E:\\program\\oceanFFT\\common\\image\\2TOP.BMP");*/
}

void SkyBoxDraw()
{
	GLfloat skyboxsize = 1.0f;
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
/**/	glBindTexture(GL_TEXTURE_2D, mTextures[0]);
	glBegin(GL_QUADS);
	glColor4ub(255, 255, 255, 255);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-skyboxsize, -skyboxsize, -skyboxsize);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(skyboxsize, -skyboxsize, -skyboxsize);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(skyboxsize, skyboxsize, -skyboxsize);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-skyboxsize, skyboxsize, -skyboxsize);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, mTextures[1]);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(skyboxsize, -skyboxsize, skyboxsize);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-skyboxsize, -skyboxsize, skyboxsize);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-skyboxsize, skyboxsize, skyboxsize);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(skyboxsize, skyboxsize, skyboxsize);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, mTextures[2]);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-skyboxsize, -skyboxsize, skyboxsize);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-skyboxsize, -skyboxsize, -skyboxsize);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-skyboxsize, skyboxsize, -skyboxsize);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-skyboxsize, skyboxsize, skyboxsize);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, mTextures[3]);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(skyboxsize, -skyboxsize, -skyboxsize);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(skyboxsize, -skyboxsize, skyboxsize);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(skyboxsize, skyboxsize, skyboxsize);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(skyboxsize, skyboxsize, -skyboxsize);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, mTextures[4]);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-skyboxsize, skyboxsize, -skyboxsize);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(skyboxsize, skyboxsize, -skyboxsize);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(skyboxsize, skyboxsize, skyboxsize);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-skyboxsize, skyboxsize, skyboxsize);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, mTextures[5]);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-skyboxsize, -skyboxsize, skyboxsize);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(skyboxsize, -skyboxsize, skyboxsize);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(skyboxsize, -skyboxsize, -skyboxsize);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-skyboxsize, -skyboxsize, -skyboxsize);
	glEnd();
	/**/
}

GLuint CreateTexture2DFromBMP(const char *bmpPath)
{
	int nFileSize = 0;
	unsigned char *bmpFileContent = LoadFileContent(bmpPath, nFileSize);
	if (bmpFileContent == nullptr) {
		return 0;
	}
	int bmpWidth = 0, bmpHeight = 0;
	unsigned char*pixelData = DecodeBMP(bmpFileContent, bmpWidth, bmpHeight);
	if (bmpWidth == 0) {
		delete bmpFileContent;
		return 0;
	}
	GLuint texture = createTexture2D(pixelData, bmpWidth, bmpHeight, GL_RGB);
	delete bmpFileContent;
	return texture;
}
unsigned char *LoadFileContent(const char *path, int &filesize)
{
	unsigned char*fileContent = nullptr;
	filesize = 0;
	FILE*pFile = fopen(path, "rb");
	if (pFile) {
		fseek(pFile, 0, SEEK_END);
		int nLen = ftell(pFile);
		if (nLen>0) {
			rewind(pFile);
			fileContent = new unsigned char[nLen + 1];
			fread(fileContent, sizeof(unsigned char), nLen, pFile);
			fileContent[nLen] = '\0';
			filesize = nLen;
		}
		fclose(pFile);
	}
	return fileContent;
}

unsigned char *DecodeBMP(unsigned char *bmpFileData, int &width, int &height)
{
	if (0x4D42 == *((unsigned short*)bmpFileData)) {
		int pixelDataOffset = *((int*)(bmpFileData + 10));
		width = *((int*)(bmpFileData + 18));
		height = *((int*)(bmpFileData + 22));
		unsigned char*pixelData = bmpFileData + pixelDataOffset;
		for (int i = 0; i<width*height * 3; i += 3) {
			unsigned char temp = pixelData[i];
			pixelData[i] = pixelData[i + 2];
			pixelData[i + 2] = temp;
		}
		return pixelData;
	}
	return nullptr;
}

GLuint createTexture2D(unsigned char *piexlData, int width, int height, GLenum type)
{
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, type, width, height, 0, type, GL_UNSIGNED_BYTE, piexlData);
	return texture;
}

/*
//glColor3f(0.0, 0.0, 0.0);  --> 黑色
//glColor3f(1.0, 0.0, 0.0);  --> 红色
//glColor3f(0.0, 1.0, 0.0);  --> 绿色
//glColor3f(0.0, 0.0, 1.0);  --> 蓝色
//glColor3f(1.0, 1.0, 0.0);  --> 黄色
//glColor3f(1.0, 0.0, 1.0);  --> 品红色
//glColor3f(0.0, 1.0, 1.0);  --> 青色
//glColor3f(1.0, 1.0, 1.0);  --> 白色

void glutWireSphere(GLdouble radius, GLint slices, GLint stacks); 线框球
void glutSolidSphere(GLdouble radius, GLint slices, GLint stacks); 实心球

void glutWireCube(GLdouble size); 线框立方体
void glutSolidCube(GLdouble size); 实心立方体

void glutWireTorus(GLdouble innerRadius, GLdouble outerRadius, GLint nsides, GLint rings); 线框圆环
void glutSolidTorus(GLdouble innerRadius, GLdouble outerRadius, GLint nsides, GLint rings); 实心圆环

void glutWireIcosahedron(void); 线框20面体
void glutSolidIcosahedron(void); 实心20面体

void glutWireOctahedron(void); 线框8面体 
void glutSolidOctahedron(void); 实心8面体

void glutWireTetrahedron(void); 线框4面体
void glutSolidTetrahedron(void); 实心4面体

void glutWireDodecahedron(GLdouble radius); 线框12面体
void glutSolidDodecahedron(GLdouble radius); 实心12面体

void glutWireCone(GLdouble radius, GLdouble height, GLint slices, GLint stacks); 线框圆锥体
void glutSolidCone(GLdouble radius, GLdouble height, GLint slices, GLint stacks); 实心圆锥体

void glutWireTeapot(GLdouble size); 线框茶壶
void glutSolidTeapot(GLdouble size); 实心茶壶


函数中，radius表示球体的半径，slices表示球体围绕z轴分割的数目，stacks表示球体沿着z轴分割的数目。

绘制中心在模型坐标原点, 半径为radius的球体, 球体围绕z轴分割slices次, 球体沿着z轴分割stacks次

GL_POINTS：把每一个顶点作为一个点进行处理，顶点n即定义了点n，共绘制N个点
GL_LINES：连接每两个顶点作为一个独立的线段，顶点2n－1和2n之间共定义了n条线段，总共绘制N/2条线段
GL_LINE_STRIP：绘制从第一个顶点到最后一个顶点依次相连的一组线段，第n和n+1个顶点定义了线段n，总共绘制n－1条线段
GL_LINE_LOOP：绘制从第一个顶点到最后一个顶点依次相连的一组线段，然后最后一个顶点和第一个顶点相连，第n和n+1个顶点定义了线段n，总共绘制n条线段
GL_TRIANGLES：把每三个顶点作为一个独立的三角形，顶点3n－2、3n－1和3n定义了第n个三角形，总共绘制N/3个三角形
GL_TRIANGLE_STRIP：绘制一组相连的三角形，对于奇数n，顶点n、n+1和n+2定义了第n个三角形；对于偶数n，顶点n+1、n和n+2定义了第n个三角形，总共绘制N-2个三角形
GL_TRIANGLE_FAN：绘制一组相连的三角形，三角形是由第一个顶点及其后给定的顶点确定，顶点1、n+1和n+2定义了第n个三角形，总共绘制N-2个三角形


glPolygonMode(GL_FRONT, GL_FILL);// 设置正面为填充模式
glPolygonMode(GL_BACK, GL_LINE);// 设置反面为线形模式
glPolygonMode(GL_BACK, GL_POINT);// 设置反面为点模式

*/