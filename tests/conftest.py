import jax

# The WCS comparisons need double precision, float32 alone is good to ~0.05"
jax.config.update("jax_enable_x64", True)
