from tables import IsDescription, Float32Col, Int8Col


class TimeSeriesDescriptor(IsDescription):
    t = Float32Col(pos=1)
    pos_x = Float32Col(pos=2)
    pos_y = Float32Col(pos=3)
    pos_z = Float32Col(pos=4)
    energy = Float32Col(pos=5)
