max_projection_shader = """
#uicontrol float gain slider(min=0, max=1000, default=100)
#uicontrol float minVal slider(min=0, max=1, default=0)
#uicontrol vec3 color color(default="white")
void main() {
  if (maxValue < minVal) {
    outputColor = vec4(vec3(0.0), 1.0);
  }
  else {
    outputColor = vec4(gain * maxValue * color, 1.0);
  }
}
"""

volume_rendering_shader = """
#uicontrol float gain slider(min=-10, max=1000)
#uicontrol invlerp normalized(range=[100, 1000], window=[0, 65535], clamp=true)
#uicontrol vec3 color color(default="white")
void main() {
  float val = normalized();
  if (VOLUME_RENDERING) {
    emitRGBA(vec4(color, val * gain));
  } else {
    emitRGB(vec3(val, val, val));
  }
}
"""
