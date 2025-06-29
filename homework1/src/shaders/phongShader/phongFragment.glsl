#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 64  // Increase from 20 to 64, drastically improves shadow quality
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

// PCSS参数定义
#define LIGHT_WORLD_SIZE 80.0     // 光源在世界空间中的大小
#define LIGHT_FRUSTUM_WIDTH 200.0 // 投影视锥体宽度
#define NEAR_PLANE 0.1            // 近平面距离
#define SEARCH_RADIUS 0.05        // 遮挡物搜索半径

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x) { 
  // -1 -1
  return fract(sin(x) * 10000.0);
}

highp float rand_2to1(vec2 uv) { 
  // 0 - 1
  const highp float a = 12.9898, b = 78.233, c = 43758.5453;
  highp float dt = dot(uv.xy, vec2(a, b)), sn = mod(dt, PI);
  return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
  const vec4 bitShift = vec4(1.0, 1.0 / 256.0, 1.0 / (256.0 * 256.0), 1.0 / (256.0 * 256.0 * 256.0));
  return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples(const in vec2 randomSeed) {

  float ANGLE_STEP = PI2 * float(NUM_RINGS) / float(NUM_SAMPLES);
  float INV_NUM_SAMPLES = 1.0 / float(NUM_SAMPLES);

  float angle = rand_2to1(randomSeed) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for(int i = 0; i < NUM_SAMPLES; i++) {
    poissonDisk[i] = vec2(cos(angle), sin(angle)) * pow(radius, 0.75);
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples(const in vec2 randomSeed) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1(randNum);
  float sampleY = rand_1to1(sampleX);

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for(int i = 0; i < NUM_SAMPLES; i++) {
    poissonDisk[i] = vec2(radius * cos(angle), radius * sin(angle));

    sampleX = rand_1to1(sampleY);
    sampleY = rand_1to1(sampleX);

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker(sampler2D shadowMap, vec2 uv, float zReceiver) {
  poissonDiskSamples(uv);

  float blockerDepthSum = 0.0;
  int blockerCount = 0;

  for(int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; i++) {
    vec2 sampleCoord = uv + poissonDisk[i] * SEARCH_RADIUS;

    // 确保采样坐标在有效范围内
    if(sampleCoord.x < 0.0 || sampleCoord.x > 1.0 ||
      sampleCoord.y < 0.0 || sampleCoord.y > 1.0) {
      continue;
    }

    vec4 depthRGBA = texture2D(shadowMap, sampleCoord);
    float sampleDepth = unpack(depthRGBA);

    // 如果采样深度小于接收点深度，说明是遮挡物
    if(sampleDepth < zReceiver - EPS) {
      blockerDepthSum += sampleDepth;
      blockerCount++;
    }
  }

  // 如果没有找到遮挡物，返回-1表示完全可见
  if(blockerCount == 0) {
    return -1.0;
  }

  // 返回遮挡物的平均深度
  return blockerDepthSum / float(blockerCount);
}

float PCF(sampler2D shadowMap, vec4 coords, float filterSize) {
  vec3 projCoords = coords.xyz / coords.w;
  projCoords = projCoords * 0.5 + 0.5;

  float currentDepth = projCoords.z;

  // Generate random seed for sampling
  vec2 randomSeed = projCoords.xy;
  poissonDiskSamples(randomSeed);
  // uniformDiskSamples(randomSeed);

  float visibility = 0.0;

  // 在圆盘滤波核内进行多次采样
  for(int i = 0; i < PCF_NUM_SAMPLES; i++) {
    vec2 sampleCoord = projCoords.xy + poissonDisk[i] * filterSize;
    vec4 depthRGBA = texture2D(shadowMap, sampleCoord);
    float sampleDepth = unpack(depthRGBA);
    if(currentDepth - EPS < sampleDepth) {
      visibility += 1.0;
    }
  }
  return visibility / float(PCF_NUM_SAMPLES); // 返回平均可见性
}

float PCSS(sampler2D shadowMap, vec4 coords) {
  // 坐标转换到投影空间
  vec3 projCoords = coords.xyz / coords.w;
  projCoords = projCoords * 0.5 + 0.5;

  // 边界检查
  if(projCoords.x < 0.0 || projCoords.x > 1.0 ||
    projCoords.y < 0.0 || projCoords.y > 1.0) {
    return 1.0;
  }
  // STEP 1: avgblocker depth
  float avgBlockerDepth = findBlocker(shadowMap, projCoords.xy, projCoords.z);
  if(avgBlockerDepth < 0.0)
    return 1.0; // 没有找到遮挡物，返回完全可见

  // STEP 2: penumbra size
  // PCSS公式: penumbra = wLight * (dReceiver - dBlocker) / dBlocker
  float penumbraSize = LIGHT_WORLD_SIZE * (projCoords.z - avgBlockerDepth) / avgBlockerDepth;
  float filterSize = penumbraSize * NEAR_PLANE / LIGHT_FRUSTUM_WIDTH;
  filterSize = clamp(filterSize, 0.001, 0.15); // 限制滤波半径，避免过大

  // STEP 3: filtering
  return PCF(shadowMap, coords, filterSize);
}

// Hard shadow effect
float useShadowMap(sampler2D shadowMap, vec4 shadowCoord) {
  vec3 projCoord = shadowCoord.xyz / shadowCoord.w;

  // Convert to [0, 1] range
  projCoord = projCoord * 0.5 + 0.5;

  vec4 depthRGBA = texture2D(shadowMap, projCoord.xy);
  float closestDepth = unpack(depthRGBA);

  float currentDepth = projCoord.z;

  float shadow = currentDepth - EPS > closestDepth ? 0.0 : 1.0;
  return shadow;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff = uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {

  float visibility;
  //visibility = useShadowMap(uShadowMap, vPositionFromLight);
  //visibility = PCF(uShadowMap, vPositionFromLight, 0.02);
  visibility = PCSS(uShadowMap, vPositionFromLight);

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);
}