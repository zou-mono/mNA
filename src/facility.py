class Facility:
    def __int__(self, id):
        self._ID = id
        self._location = None
        self._name = ""
        self._level = 10000  # 等级越大越好
        self._capacity = -1

    @property
    def ID(self):
        return self._ID

    @ID.setter
    def ID(self, v):
        self._ID = v

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, v):
        self._name = v

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, v):
        self._location = v

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, v):
        self._level = v

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, v):
        self._capacity = v
