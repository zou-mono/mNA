
class Person:
    def __int__(self, id):
        self._age = None
        self._facility = None
        self._ID = id
        self._location = None
        self._facility_order = None  # 设施在最终计算List中的序号

    @property
    def ID(self):
        return self._ID

    @ID.setter
    def ID(self, v):
        self._ID = v

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, v):
        self._age = v

    @property
    def facility(self):
        return self._facility

    @facility.setter
    def facility(self, v):
        self._facility = v

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, v):
        self._location = v

    @property
    def facility_order(self):
        return self._facility_order

    @facility_order.setter
    def facility_order(self, v):
        self._facility_order = v

