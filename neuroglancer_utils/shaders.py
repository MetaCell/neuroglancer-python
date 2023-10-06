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

# NOTE: not a real isosurface, just to see what this kind of GPU-based isosurface
# would look like in neuroglancer with the different chunks etc.
iso_surface_shader = """
#uicontrol float gain slider(min=-10, max=1000)
#uicontrol invlerp normalized(range=[100, 1000], window=[0, 65535], clamp=true)
#uicontrol vec3 color color(default="white")
#uicontrol float inttargetmin slider(min=0, max=65535)
#uicontrol float inttargetmax slider(min=0, max=65535)
void main() {
  float targetmin = inttargetmin / 65535.0;
  float targetmax = inttargetmax / 65535.0;
  float val = normalized();
  if (val >= targetmin && val <= targetmax) {
    outputColor = (vec4(color * gain, 1.0));
    //TODO would break; in real example
  }
}
"""
