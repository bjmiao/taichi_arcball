
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_theta(self, theta, x, y, z):
        self.w = ti.sin(theta)
        cos_theta = ti.cos(theta)
        self.x = cos_theta * x
        self.y = cos_theta * y
        self.z = cos_theta * z
    def __str__(self):
        return f"({self.w:.2f} {self.x:.2f} {self.y:.2f} {self.z:.2f})"
 
    def prod(self, rhs):
        w = self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z
        x = self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y
        y = self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x
        z = self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w
        return Quaternion(w, x, y, z)

    @staticmethod
    def from_diagonal_matrix(mat: ti.Matrix):
        w = ti.sqrt(1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]) / 2.0
        w4 = 4.0 * w
        x = (mat[2, 1] - mat[1, 2]) / w4
        y = (mat[0, 2] - mat[2, 0]) / w4
        z = (mat[1, 0] - mat[0, 1]) / w4
        return Quaternion(w, x, y, z)
        # w = Math.sqrt(1.0 + m1.m00 + m1.m11 + m1.m22) / 2.0;
        # double w4 = (4.0 * w);
        # x = (m1.m21 - m1.m12) / w4 ;
        # y = (m1.m02 - m1.m20) / w4 ;
        # z = (m1.m10 - m1.m01) / w4 ;
    def norm(self):
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z 

    def get_elements(self):
        return self.x, self.y, self.z
    # @classmethod
    # def multiply(i: Quaternion, j: Quaternion):
    #     pass
