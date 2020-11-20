from tables import IsDescription, Float32Col, ComplexCol


class TransformsDescriptor(IsDescription):
    energy = Float32Col(pos=1)
    frequency = Float32Col(pos=2)
    I_xx = ComplexCol(pos=3)
    I_yy = ComplexCol(pos=4)
    I_zz = ComplexCol(pos=5)
