// GLSL vertex shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
varying vec4 pos;
varying float scaledHeight;

uniform float heightScale;
uniform float chopiness;
uniform vec2 size;


void main()
{
    float height = gl_MultiTexCoord0.x;
    scaledHeight = height * heightScale;
    vec2 slope = gl_MultiTexCoord1.xy;

    // calculate surface normal from slope for shading
    vec3 normal = normalize(cross(vec3(0.0, slope.y * heightScale, 2.0 / size.x), vec3(2.0 / size.y, slope.x * heightScale, 0.0)));
    worldSpaceNormal = normal;

    // calculate position and transform to homogeneous clip space
    pos = vec4(gl_Vertex.x, scaledHeight, gl_Vertex.z, 1.0);
    gl_Position = gl_ModelViewProjectionMatrix * pos;

    eyeSpacePos = (gl_ModelViewMatrix * pos).xyz;
    eyeSpaceNormal = (gl_NormalMatrix * normal).xyz;
}
