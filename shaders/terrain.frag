// GLSL fragment shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
varying vec4 pos;
varying float height;

uniform vec4 waterColor;
uniform vec4 landColor;
uniform vec4 mountainColor;
uniform vec4 skyColor;
uniform vec3 lightDir;

void main()
{
    vec3 eyeVector = normalize(eyeSpacePos);
    vec3 eyeSpaceNormalVector = normalize(eyeSpaceNormal);
    vec3 worldSpaceNormalVector = normalize(worldSpaceNormal);

    float facing = max(0.0, dot(eyeSpaceNormalVector, -eyeVector));
    float diffuse = max(0.0, dot(worldSpaceNormalVector, lightDir));

    vec4 selectedColor;

    if (height <= 0.3) {
        selectedColor = waterColor;
    } else if (height <= 0.7) {
        selectedColor = landColor;
    } else if (height <= 0.75) {
        float pct = 20 * (height - 0.7);
        selectedColor = mix(landColor, mountainColor, pct);
    } else {
        selectedColor = mountainColor;
    }
    gl_FragColor = selectedColor * diffuse;
}