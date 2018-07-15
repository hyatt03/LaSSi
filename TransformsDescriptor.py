from tables import IsDescription, Float32Col


class TransformsDescriptor(IsDescription):
    energy = Float32Col()
    frequency = Float32Col()
    I_xx = Float32Col()
    I_yy = Float32Col()
    I_zz = Float32Col()
