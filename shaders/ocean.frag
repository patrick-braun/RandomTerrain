// GLSL fragment shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
varying float scaledHeight;

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
    float fresnel = pow(1.0 - facing, 5.0); // Fresnel approximation
    float diffuse = max(0.0, dot(worldSpaceNormalVector, lightDir));

    vec4 selectedColor;

    if (scaledHeight < 0.01) {
        selectedColor = waterColor;
    } else if (scaledHeight <= 0.2) {
        selectedColor = landColor;
    } else if (scaledHeight <= 0.25) {
        float pct = (1.0 / 0.05) * (scaledHeight - 0.2);
        selectedColor = mix(landColor, mountainColor, pct);
    } else {
        selectedColor = mountainColor;
    }


    gl_FragColor = selectedColor * diffuse + skyColor * fresnel;
}