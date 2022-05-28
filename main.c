/*
    James William Fletcher (james@voxdsp.com)
        May 2022

    Info:
    
        A hover car simulation on mars.
        
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/file.h>
#include <stdint.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/time.h>

#define uint GLuint
#define f32 GLfloat

#include "inc/gl.h"
#define GLFW_INCLUDE_NONE
#include "inc/glfw3.h"

#ifndef __x86_64__
    #define NOSSE
#endif

// uncommenting this define will enable the MMX random when using fRandFloat (it's a marginally slower)
#define SEIR_RAND

#include "inc/esAux2.h"

#include "inc/res.h"

#include "assets/low.h"
#include "assets/inner.h"
#include "assets/outer.h"
#include "assets/hova.h"

//*************************************
// globals
//*************************************
GLFWwindow* window;
uint winw = 1024;
uint winh = 768;
double t = 0;   // time
f32 dt = 0;     // delta time
double fc = 0;  // frame count
double lfct = 0;// last frame count time
double lc = 0;  // logic count
double llct = 0;// last logic count time
f32 aspect;
double x,y,lx,ly;
double rww, ww, rwh, wh, ww2, wh2;
double uw, uh, uw2, uh2; // normalised pixel dpi

// render state id's
GLint projection_id;
GLint modelview_id;
GLint position_id;
GLint lightpos_id;
GLint solidcolor_id;
GLint color_id;
GLint opacity_id;
GLint normal_id; 

// render state matrices
mat projection;
mat view;
mat model;
mat modelview;

// render state inputs
vec lightpos = {0.f, 0.f, 0.f};

// models
ESModel mdlSphere;
ESModel mdlInner;
ESModel mdlOuter;
ESModel mdlHova;

// sim vars
#define FAR_DISTANCE 160.f
uint RENDER_PASS = 0;
double st=0; // start time
char tts[32];// time taken string
const f32 simspeed = 0.1f;

// cosmos
#define COSMOS_SIZE 256
vec cosmos[COSMOS_SIZE];
vec cosmos_color[COSMOS_SIZE];
f32 cosmos_scale[COSMOS_SIZE];

// hova sim
f32 th = 14.5f; // terrain height
f32 hh = 14.3f; // hova height

//*************************************
// utility functions
//*************************************
void timestamp(char* ts)
{
    const time_t tt = time(0);
    strftime(ts, 16, "%H:%M:%S", localtime(&tt));
}

static inline float fRandFloat(const float min, const float max)
{
    return min + randf() * (max-min); 
}

void timeTaken(uint ss)
{
    if(ss == 1)
    {
        const double tt = t-st;
        if(tt < 60.0)
            sprintf(tts, "%.0f Sec", tt);
        else if(tt < 3600.0)
            sprintf(tts, "%.2f Min", tt * 0.016666667);
        else if(tt < 216000.0)
            sprintf(tts, "%.2f Hr", tt * 0.000277778);
        else if(tt < 12960000.0)
            sprintf(tts, "%.2f Days", tt * 0.00000463);
    }
    else
    {
        const double tt = t-st;
        if(tt < 60.0)
            sprintf(tts, "%.0f Seconds", tt);
        else if(tt < 3600.0)
            sprintf(tts, "%.2f Minutes", tt * 0.016666667);
        else if(tt < 216000.0)
            sprintf(tts, "%.2f Hours", tt * 0.000277778);
        else if(tt < 12960000.0)
            sprintf(tts, "%.2f Days", tt * 0.00000463);
    }
}

float urandf()
{
    static const float FLOAT_UINT64_MAX = (float)UINT64_MAX;
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return (((float)s)+1e-7f) / FLOAT_UINT64_MAX;
}

float uRandFloat(const float min, const float max)
{
    return ( urandf() * (max-min) ) + min;
}

uint64_t urand()
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return s;
}

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

//*************************************
// render functions
//*************************************

__attribute__((always_inline)) inline void modelBind(const ESModel* mdl) // C code reduction helper (more inline opcodes)
{
    glBindBuffer(GL_ARRAY_BUFFER, mdl->vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdl->iid);
}

__attribute__((always_inline)) inline void modelBind3(const ESModel* mdl) // C code reduction helper (more inline opcodes)
{
    glBindBuffer(GL_ARRAY_BUFFER, mdl->cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdl->vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdl->nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdl->iid);
}

void rSphere(f32 x, f32 y, f32 z, f32 s)
{
    mIdent(&model);
    mTranslate(&model, x, y, z);
    mScale(&model, s, s, s);
    mMul(&modelview, &model, &view);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    glDrawElements(GL_TRIANGLES, low_numind, GL_UNSIGNED_SHORT, 0);
}

//*************************************
// sim functions
//*************************************

void newSim()
{
    const int seed = urand();
    srand(seed);
    srandf(seed);

    st = t;

    char strts[16];
    timestamp(&strts[0]);
    printf("[%s] Sim Start [%u].\n", strts, seed);
    
    glfwSetWindowTitle(window, "Mars Racer");
    
    for(uint i = 0; i < COSMOS_SIZE; i++)
    {
        vRuvBT(&cosmos[i]); // random point on outside of unit sphere
        cosmos_color[i].x = randf();
        cosmos_color[i].y = randf();
        cosmos_color[i].z = randf();
        vMulS(&cosmos[i], cosmos[i], fRandFloat(32.f, 72.f));
        cosmos_scale[i] = fRandFloat(0.1f, 1.6f);
    }
}

//*************************************
// update & render
//*************************************
void main_loop()
{
//*************************************
// update title bar stats
//*************************************
    static double ltut = 3.0;
    if(t > ltut)
    {
        timeTaken(1);
        char title[512];
        sprintf(title, "| %s |", tts);
        glfwSetWindowTitle(window, title);
        ltut = t + 1.0;
    }

//*************************************
// camera
//*************************************
    if(RENDER_PASS == 1)
    {
        mIdent(&view);
        mRotY(&view, 234.f*DEG2RAD);
        mTranslate(&view, 0.f, 0.f, th + (fabsf(sinf(t*simspeed))*0.1f) + 0.2f); //th + 0.2f
        mRotY(&view, -t*simspeed);

//*************************************
// render
//*************************************
        // clear render and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // prep mars for rendering
        shadeLambert3(&position_id, &projection_id, &modelview_id, &lightpos_id, &normal_id, &color_id, &opacity_id);
        glUniform3f(lightpos_id, lightpos.x, lightpos.y, lightpos.z);
        glUniform1f(opacity_id, 1.0f);

        // maticies
        glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &view.m[0][0]);

        // render
        modelBind3(&mdlOuter);
        glDisable(GL_CULL_FACE); // because I messed up in blender.. i know.. damn. I know not to do that next time.
        glDrawElements(GL_TRIANGLES, outer_numind, GL_UNSIGNED_INT, 0);
        glEnable(GL_CULL_FACE);
        modelBind3(&mdlInner);
        glDrawElements(GL_TRIANGLES, inner_numind, GL_UNSIGNED_INT, 0);

        // render cosmos
        shadeLambert(&position_id, &projection_id, &modelview_id, &lightpos_id, &color_id, &opacity_id);
        glUniform3f(lightpos_id, lightpos.x, lightpos.y, lightpos.z);
        glUniform1f(opacity_id, 1.0f);
        modelBind(&mdlSphere);
        for(uint i = 0; i < COSMOS_SIZE; i++)
        {
            glUniform3f(color_id, cosmos_color[i].x, cosmos_color[i].y, cosmos_color[i].z);
            rSphere(cosmos[i].x, cosmos[i].y, cosmos[i].z, cosmos_scale[i]);
        }

        // begin hova rendering
        shadeLambert3(&position_id, &projection_id, &modelview_id, &lightpos_id, &normal_id, &color_id, &opacity_id);
        glUniform3f(lightpos_id, lightpos.x, lightpos.y, lightpos.z);
        glUniform1f(opacity_id, 1.0f);

        // prep matrix
        mIdent(&model);
        mRotY(&model, (t*simspeed)+0.017f); // camtracking rot
        mRotX(&model, 180.f*DEG2RAD);
        mTranslate(&model, sinf(t*simspeed)*0.1f, 0.f, 14.3f); // 14.3f is the terrain midpoint we will sample for height offset correction

        // workout average terrain height (will be 1 frame lagging due to order of exec, who cares.)
        vec pos;
        mGetPos(&pos, model);
        f32 ah = 0.f;
        f32 ahc = 0.f;
        const uint vas = ((uint)inner_numvert)*3;
        for(uint i = 0; i < vas; i+=3)
        {
            vec vp;
            vp.x = inner_vertices[i];
            vp.y = inner_vertices[i+1];
            vp.z = inner_vertices[i+2];
            if(vDist(vp, pos) < 0.63f) // 0.63f is the average sample range
            {
                ah += vMod(vp);
                ahc += 1.f;
            }
        }
        if(ahc > 0.f)
        {
            ah /= ahc;
            if(ah > th) // only adjust up and let gravity pull down
                th += simspeed*88.f*(ah-th)*dt; // thrust th = ah;
        }

        // "mars gravity"
        th -= simspeed*dt;

        // stat to console
        //printf("%f %f %f %f\n", pos.x, pos.y, pos.z, ah);

        // correct height
        mTranslate(&model, 0.f, 0.f, (th-14.3f)+0.16f);

        // make modelview
        mMul(&modelview, &model, &view);

        // set matricies for shader
        glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);

        // render hova
        modelBind3(&mdlHova);
        glDrawElements(GL_TRIANGLES, hova_numind, GL_UNSIGNED_SHORT, 0);

        // dislay new render
        glfwSwapBuffers(window);
    }
}

//*************************************
// Input Handelling
//*************************************
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // control
    if(action == GLFW_PRESS)
    {
        // new
        if(key == GLFW_KEY_N)
        {
            // end
            timeTaken(0);
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] Sim End.\n", strts);
            printf("[%s] Time-Taken: %s or %g Seconds\n\n", strts, tts, t-st);
            
            // new
            newSim();
        }

        // show average fps
        else if(key == GLFW_KEY_F)
        {
            if(t-lfct > 2.0)
            {
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] FPS: %g\n", strts, fc/(t-lfct));
                printf("[%s] LPS: %g\n", strts, lc/(t-llct));
                lfct = t;
                fc = 0;
                llct = t;
                lc = 0;
            }
        }
    }
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    winw = width;
    winh = height;

    glViewport(0, 0, winw, winh);
    aspect = (f32)winw / (f32)winh;
    ww = winw;
    wh = winh;
    rww = 1/ww;
    rwh = 1/wh;
    ww2 = ww/2;
    wh2 = wh/2;
    uw = (double)aspect / ww;
    uh = 1 / wh;
    uw2 = (double)aspect / ww2;
    uh2 = 1 / wh2;

    mIdent(&projection);
    mPerspective(&projection, 90.0f, aspect, 0.01f, FAR_DISTANCE*2.f); 
}

//*************************************
// Process Entry Point
//*************************************
int main(int argc, char** argv)
{
    // allow custom msaa level
    int msaa = 16;
    if(argc >= 2){msaa = atoi(argv[1]);}

    // allow framerate cap
    double maxfps = 144.0;
    if(argc >= 3){maxfps = atof(argv[2]);}

    // help
    printf("----\n");
    printf("Mars Racer\n");
    printf("----\n");
    printf("James William Fletcher (james@voxdsp.com)\n");
    printf("----\n");
    printf("Argv(2): msaa, maxfps\n");
    printf("e.g; ./uc 16 60\n");
    printf("----\n");
    printf("N = New sim.\n");
    printf("F = FPS to console.\n");
    printf("----\n");

    // init glfw
    if(!glfwInit()){printf("glfwInit() failed.\n"); exit(EXIT_FAILURE);}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SAMPLES, msaa);
    window = glfwCreateWindow(winw, winh, "Mars Racer", NULL, NULL);
    if(!window)
    {
        printf("glfwCreateWindow() failed.\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    const GLFWvidmode* desktop = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (desktop->width/2)-(winw/2), (desktop->height/2)-(winh/2)); // center window on desktop
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetKeyCallback(window, key_callback);
    //glfwSetMouseButtonCallback(window, mouse_button_callback);
    //glfwSetScrollCallback(window, scroll_callback);
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(0); // 0 for immediate updates, 1 for updates synchronized with the vertical retrace, -1 for adaptive vsync

    // set icon
    glfwSetWindowIcon(window, 1, &(GLFWimage){16, 16, (unsigned char*)&icon_image.pixel_data});

    // hide cursor
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

//*************************************
// projection
//*************************************

    window_size_callback(window, winw, winh);

//*************************************
// bind vertex and index buffers
//*************************************

    // ***** BIND SPHERE *****
    esBind(GL_ARRAY_BUFFER, &mdlSphere.vid, low_vertices, sizeof(low_vertices), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlSphere.iid, low_indices, sizeof(low_indices), GL_STATIC_DRAW);

    // ***** BIND INNER *****
    esBind(GL_ARRAY_BUFFER, &mdlInner.cid, inner_colors, sizeof(inner_colors), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlInner.vid, inner_vertices, sizeof(inner_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlInner.nid, inner_normals, sizeof(inner_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlInner.iid, inner_indices, sizeof(inner_indices), GL_STATIC_DRAW);

    // ***** BIND OUTER *****
    esBind(GL_ARRAY_BUFFER, &mdlOuter.cid, outer_colors, sizeof(outer_colors), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlOuter.vid, outer_vertices, sizeof(outer_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlOuter.nid, outer_normals, sizeof(outer_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlOuter.iid, outer_indices, sizeof(outer_indices), GL_STATIC_DRAW);

    // ***** BIND HOVA *****
    esBind(GL_ARRAY_BUFFER, &mdlHova.cid, hova_colors, sizeof(hova_colors), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlHova.vid, hova_vertices, sizeof(hova_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlHova.nid, hova_normals, sizeof(hova_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlHova.iid, hova_indices, sizeof(hova_indices), GL_STATIC_DRAW);

//*************************************
// compile & link shader programs
//*************************************

    //makeAllShaders();
    makeLambert();
    makeLambert3();

//*************************************
// configure render options
//*************************************

    // glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    // glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 0.0f);

//*************************************
// execute update / render loop
//*************************************

    // init
    const double maxlps = 60.0;
    t = glfwGetTime();
    lfct = t;
    dt = 1.0 / (float)maxlps; // fixed timestep delta-time
    newSim();
    
    // lps accurate event loop
    const double fps_limit = 1.0 / maxfps;
    double rlim = 0.0;
    const useconds_t wait_interval = 1000000 / maxlps; // fixed timestep
    useconds_t wait = wait_interval;
    while(!glfwWindowShouldClose(window))
    {
        usleep(wait);
        t = glfwGetTime();
        glfwPollEvents();

        if(maxfps < maxlps)
        {
            if(t > rlim)
            {
                RENDER_PASS = 1;
                rlim = t + fps_limit; // should be doing this after main_loop() at the very least but it's not a big deal.
                fc++;
            }
            else
                RENDER_PASS = 0;
        }
        else
        {
            RENDER_PASS = 1;
            fc++;
        }

        main_loop();

        // accurate lps
        wait = wait_interval - (useconds_t)((glfwGetTime() - t) * 1000000.0);
        if(wait > wait_interval)
            wait = wait_interval;
        lc++;
    }

    // end
    timeTaken(0);
    char strts[16];
    timestamp(&strts[0]);
    printf("[%s] Sim End.\n", strts);
    printf("[%s] Time-Taken: %s or %g Seconds\n\n", strts, tts, t-st);

    // done
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
    return 0;
}
