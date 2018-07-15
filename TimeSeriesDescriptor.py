from tables import IsDescription, Float32Col, Int8Col


class TimeSeriesDescriptor(IsDescription):
    t = Float32Col()
    pos_x = Float32Col()
    pos_y = Float32Col()
    pos_z = Float32Col()
    energy = Float32Col()
