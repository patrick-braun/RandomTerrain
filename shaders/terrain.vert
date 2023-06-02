// GLSL vertex shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
varying vec4 pos;
varying float height;

uniform vec2 size;


void main()
{
    height = gl_MultiTexCoord0.x;
    vec2 slope = gl_MultiTexCoord1.xy;

    // calculate surface normal from slope for shading
    vec3 normal = normalize(cross(vec3(0.0, slope.y, 2.0 / size.x), vec3(2.0 / size.y, slope.x, 0.0)));
    worldSpaceNormal = normal;

    // calculate position and transform to homogeneous clip space
    pos = vec4(gl_Vertex.x, height, gl_Vertex.z, 1.0);
    gl_Position = gl_ModelViewProjectionMatrix * pos;

    eyeSpacePos = (gl_ModelViewMatrix * pos).xyz;
    eyeSpaceNormal = (gl_NormalMatrix * normal).xyz;
}
