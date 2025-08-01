class DirectionalLight {
  constructor(
    lightIntensity,
    lightColor,
    lightPos,
    focalPoint,
    lightUp,
    hasShadowMap,
    gl
  ) {
    this.mesh = Mesh.cube(setTransform(0, 0, 0, 0.2, 0.2, 0.2, 0));
    this.mat = new EmissiveMaterial(lightIntensity, lightColor);
    this.lightPos = lightPos;
    this.focalPoint = focalPoint;
    this.lightUp = lightUp;

    this.hasShadowMap = hasShadowMap;
    this.fbo = new FBO(gl);
    if (!this.fbo) {
      console.log("无法设置帧缓冲区对象");
      return;
    }
  }

  CalcLightMVP(translate, scale) {
    let lightMVP = mat4.create();
    let modelMatrix = mat4.create();
    let viewMatrix = mat4.create();
    let projectionMatrix = mat4.create();

    // Model transform
    mat4.translate(modelMatrix, modelMatrix, translate);
    mat4.scale(modelMatrix, modelMatrix, scale);

    // View transform
    mat4.lookAt(viewMatrix, this.lightPos, this.focalPoint, this.lightUp);

    // Projection transform - Use orthographic projection for linearity of depth
    const left = -100;
    const right = -left;
    const bottom = -100;
    const top = -bottom;
    const near = 0.1;
    const far = 400;
    mat4.ortho(projectionMatrix, left, right, bottom, top, near, far);

    mat4.multiply(lightMVP, projectionMatrix, viewMatrix);
    mat4.multiply(lightMVP, lightMVP, modelMatrix);

    return lightMVP;
  }
}
