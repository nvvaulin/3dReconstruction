import cv2
import numpy as np
import math
from math import sin
from math import cos
from numpy.linalg import inv
from numpy.random import random
import matplotlib.pyplot as plt

class Vertex(object):
    def __init__(self, id, t, r):
        self.id = id
        self.t = np.float64(t)
        self.r = np.float64(r)
        self.e = {}
        self.M = None
        self.trueR = None
        self.trueT = None

    def __getitem__(self, item):
        return self.e[item]

    def addE(self, vertex):
        self.e.update({ vertex :[] })

    def points(self,item):
        p = []
        for i in self.e[item]:
            p.append([i.p[0,0],i.p[1,0]])
        return np.float64(p)

class Edge(object):
    def __init__(self, p, K, vertex):
        self.v = vertex
        self.p = p
        self.Kp = K.dot(self.p)
        self.Kp /= math.sqrt(np.dot(np.transpose(self.Kp),self.Kp))
        self.n = None
        self.trueP = None


class Graph(object):
    def __init__(self,arp, im_size):
        self.E = []
        self.V = []
        self.arp = arp
        self.im_size = np.array([im_size[0],im_size[1]])
        self.K = np.mat(np.ndarray((3,3),dtype=np.float))
        self.K[...] = 0
        for i in range(2):
            self.K[i, i] = -2*math.tan(arp[i]/2)/im_size[i]
            self.K[i, 2] = math.tan(arp[i]/2)
        self.K[2,2] = 1

    #tested
    def rotatedRect(self, v):
        '''
        :param v:
        :return: ((t.x,t.y), (w,h), angle in degree) in global coordinates
        '''
        s = [2*math.tan(self.arp[i]/2)*v.t[2] for i in range(2)]
        return ((v.t[0],v.t[1]),(s[0],s[1]), v.r[2]*180.0/np.pi)

    #tested
    def rotatedRectToPoints(self, r):
        M = cv2.getRotationMatrix2D(r[0],-r[2],1)
        w = r[1][0]/2
        h = r[1][1]/2
        M[:,2] = r[0]
        points = np.array([[w,h,1],[w,-h,1],[-w,-h,1],[-w,h,1]])
        return self.affineTransform(points,M)

    #tested
    def intersection(self, v1,v2):
        inter = cv2.rotatedRectangleIntersection(self.rotatedRect(v1),self.rotatedRect(v2))
        if inter is None:
            return None
        if inter[1] is None:
            return None
        inter = np.array( [i[0,:] for i in inter[1]])
        inter = cv2.convexHull(cv2.rotatedRectangleIntersection(self.rotatedRect(v1),self.rotatedRect(v2))[1])
        inter = np.array( [i[0] for i in inter])
        return inter

    #tested
    def findE(self):
        for i in range(len(self.V)-1):
            for j in range(i+1, len(self.V)):
                inter = self.intersection(self.V[i],self.V[j])
                if not(inter is None):
                    self.V[i].addE(self.V[j])
                    self.V[j].addE(self.V[i])
                    self.E.append((self.V[i],self.V[j]))

    def getAffine(self, f, t  ):
        '''
        :param f: vertex from
        :param t: vertex to
        :return: M[2x3]
        '''
        r = (f.t-t.t)[:2]*np.array([self.im_size[i]/(2*math.tan(self.arp[i]/2)*t.t[2]) for i in range(2)])
        M = cv2.getRotationMatrix2D((self.im_size[0]/2,self.im_size[1]/2),(t.r[2]-f.r[2])*180.0/np.pi, f.t[2]/t.t[2])
        M[:,2] = r
        return M

    def affineTransform(self,p,M):
        if len(p[0]) < 3 :
            p = np.array([[i[0],i[1],1] for i in p])
        return np.array([M.dot(i) for i in p])

    #tested
    def addV(self,id,  pos,rot):
        self.V.append(Vertex(id,pos,rot))

    #tested
    def drawV(self, color = (0,255,0)):
        for v in self.V:
            p = self.rotatedRectToPoints(self.rotatedRect(v))
            cv2.polylines(self.img, np.int32([p]),True,color)

    #tested
    def drawE(self, color = (255,0,0) ):
        for e in self.E:
            cv2.polylines(self.img, np.int32([[e[0].t[:2],e[1].t[:2]]]),False,color)
            cv2.polylines(self.img, np.int32([self.intersection(e[0],e[1])]),True,color,1)

    #tested
    def initTest(self, n, m, inersectionK=0.5, noise_p = 0.0, noise_r = 0.0):
        self.img = np.ndarray((1000,1000,3),dtype = 'uint8')
        self.img[...] = 255
        step = max(self.img.shape[1]/(n),self.img.shape[0]/(m))
        h = step/(2*math.tan(self.arp[0]/2.0)*inersectionK)
        for i in range(n):
            for j in range(m):
                t = np.array([step*(i+0.5)+step/6.0*random(),step*(j+0.5)+step/6.0*random(),h + 0.05*h*(0.5+random())])
                r = np.array([0, 0, (random()-0.5)*np.pi])
                self.addV(i*m+j, t, r)
        for v in self.V:
            v.trueT = v.t
            v.trueR = v.r

    #tested
    def generatePoints(self,keys_on_image, delta_h=0 ):
        for e in self.E:
            cnt = self.intersection(e[0],e[1])
            rectPoints = self.rotatedRectToPoints(self.rotatedRect(e[0]))
            rect = cv2.boundingRect(np.float32(rectPoints))
            count = 0
            while count < keys_on_image:
                p = np.array([rect[0]+random()*rect[2], rect[1]+random()*rect[3],-delta_h*e[0].t[2]*(0.5+random())])
                point = (int(p[0]),int(p[1]))
                if cv2.pointPolygonTest(np.float32(rectPoints), point, False) > 0:
                    count += 1
                    if cv2.pointPolygonTest(np.float32(cnt), point, False) > 0:
                        p0 = self.projectPoint(p, e[0])
                        p1 = self.projectPoint(p, e[1])
                        e[0][e[1]].append(Edge(np.transpose(p0),self.K,e[0]))
                        e[1][e[0]].append(Edge(np.transpose(p1),self.K,e[1]))
                        e[1][e[0]][-1].trueP = p
                        e[0][e[1]][-1].trueP = p

    def addNoiseToPositions(self, noise_p, noise_h, noise_r, noise_rz):
        for v in self.V:
            v.t += np.array([noise_p*random(),noise_p*random(),noise_h*random()])
            v.r += np.array([noise_r*random(),noise_r*random(),noise_rz*random()])

    #tested
    def drawTruePoints(self, color = (255,0,0)):
        for e in self.E:
            for p in e[0][e[1]]:
                cv2.circle(self.img, (int(p.trueP[0]),int(p.trueP[1])),2,color,-1)

    #tested
    def drawPoints(self, color = (0,0,255)):
        for e in self.E:
            for i in range(len(e[0][e[1]])):
                p  = self.getPointPosition(e[0][e[1]][i],e[1][e[0]][i])
                cv2.circle(self.img, (int(p[0,0]),int(p[0,1])),2,color,-1)

    #tested
    def projectPoint(self, point, v):
        p = inv(self.K).dot(inv(self.rotation(v.r)).dot(point-v.t))
        p /= p[0, 2]
        return p

    #tested
    def getPointPosition(self, p1,p2):
        n1 = self.lineDirection(p1, self.rotation(p1.v.r))
        n2 = self.lineDirection(p2, self.rotation(p2.v.r))
        dist =self.distance(n1,n2,p1.v.t,p2.v.t)
        if dist[0] > 0.01:
            print dist[0]
        return ((n1*dist[1]+p1.v.t)+(n2*dist[2]+p2.v.t))/2.0

    #tested
    @staticmethod
    def distance(n1, n2, r1, r2):
        mul = np.mat(np.cross(n1, n2)[0, :])
        r = np.mat(r2-r1)
        a1 = np.dot(r, np.transpose(n1))[0,0]
        if np.dot(mul, np.transpose(mul)) < 0.0025:
            r -= n1*a1
            return np.array([math.sqrt(np.dot(r,np.transpose(r))[0,0]),0,0], dtype=np.float)
        else:
            a2 = np.dot(r,np.transpose(n2))[0, 0]
            alpha = np.dot(n1,np.transpose(n2))[0, 0]
            sq_alpha = 1.0-alpha*alpha
            t1 = (a1 - a2*alpha)/sq_alpha
            t2 = -(a2 - a1*alpha)/sq_alpha
            return np.array([np.dot(mul, np.transpose(r))[0, 0], t1, t2], dtype=np.float)

    #tested
    @staticmethod
    def rotation(r):
        Rx = np.array([[1,0,0],[0, cos(r[0]), -sin(r[0])],[0, sin(r[0]), cos(r[0])]])
        Ry = np.array([[cos(r[1]), 0, -sin(r[1])],[0,1,0],[sin(r[1]), 0, cos(r[1])]])
        Rz = np.array([[cos(r[2]), -sin(r[2]), 0],[sin(r[2]),  cos(r[2]), 0],[0,0,1]])
        return Rz.dot(Ry.dot(Rx))

    #tested
    def lineDirection(self, edge, M):
        return np.transpose(M.dot(edge.Kp))

    def preCalc(self):
        for v in self.V:
            v.M = self.rotation(v.r)
        for e in self.E:
            e.n = self.lineDirection(e,e.v.M)

    @staticmethod
    def normAngle(a):
        return np.array([((i+np.pi) - float(int((i+np.pi)/(2*np.pi)))*2*np.pi)-np.pi for i in a])

    def errorRotation(self, src, dst):
        ps = src.points(dst)
        ps = np.array([ i for i in ps])
        pd = dst.points(src)
        E = cv2.findEssentialMat(ps, pd)
        if E is None:
            return
        try:
            decomposition = cv2.decomposeEssentialMat(E[0])
        except:
            print E
        r1 = (self.decomposeRotation(inv(self.rotation(src.r)).dot(decomposition[0].dot(self.rotation(dst.r))))+np.pi/2) % np.pi - np.pi/2
        r2 = (self.decomposeRotation(inv(self.rotation(src.r)).dot(decomposition[1].dot(self.rotation(dst.r))))+np.pi/2) % np.pi - np.pi/2
        if r1.dot(np.transpose(r1)) < r2.dot(np.transpose(r2)):
            return r1,r2,E[1]
        else:
            return r1,r2,E[1]

    #tested
    def deleteWeekEdges(self):
        E = []
        for e in self.E:
            if len(e[0].e[e[1]]) > 4:
                E.append(e)
            else:
                del e[0].e[e[1]]
                del e[1].e[e[0]]
        self.E = E

    def getMeanRotations(self, v):
        rot = []
        w = []
        for dst in v.e:
            r = self.recoverRotation(v,dst)
            if not (r is None):
                rot.append(r[0])
                w.append(r[1].sum())
        if len(rot) == 0:
            return np.array([0.0,0.0,0.0])

        lr = [ np.transpose(i).dot(i)[0,0] for i in rot]
        sorted(lr)
        m = lr[len(lr)/2]*5
        sw = 0
        r = np.array([0.0,0.0,0.0])
        for i in range(len(rot)):
            if(np.transpose(rot[i]).dot(rot[i])[0,0] < m):
                sw+=w[i]
                r += rot[i]*w[i]
        return r/sw

    def optimiseRotation(self):
        V = dict([ (v,[]) for v in self.V])
        rot = dict([(v,np.array([0.0,0.0,0.0])) for v in self.V])
        c = 0.01
        for e in self.E:
            r = self.errorRotation(e[0],e[1])
            if not (r is None):
                V[e[0]].append((e[1],r[0],r[1].sum()))
                V[e[1]].append((e[0],-r[0],r[1].sum()))
        Vt = []
        for v in V.items():
            t = [ math.sqrt(np.dot(i[1],i[1])) for i in v[1]]
            sorted(t)
            m = t[len(t)/2]*10
            edges = []
            for i in v[1]:
                if np.dot(i[1],i[1]) < m*m:
                    edges.append(i)
            Vt.append((v[0],edges))
        V = dict(Vt)

        mean_mist = []
        for i in range(100):
            #for debug
            mist = []

            #r^i_(n+1)  = (1-c) * sum( (r^j_n+r_(i,j))*w_(i,j) )/sum(w_(i,j)) - c*r_i^0
            shuffeled = range(len(self.V))
            np.random.shuffle(shuffeled)
            for id in shuffeled:
                v = self.V[id]
                e = V[v]
                sum = 0
                s = 0
                for i in e:
                    sum += (rot[i[0]]+i[1])*i[2]
                    s+=i[2]
                sum /= s
                rot[v] = sum
                mist.append(math.sqrt(np.dot(v.trueR + rot[v]-v.r, v.trueR + rot[v]-v.r)))
            mean_mist.append(np.array(mist).mean())

        plt.plot(mean_mist)
        plt.show()


    #tested
    @staticmethod
    def decomposeRotation(R):
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(R[2,0], math.sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2]))
        z = math.atan2(R[1,0], R[0,0])
        return np.array([x,y,z])

g = Graph((np.pi/2,np.pi/2),(50,50))
g.initTest(4,4,1)
g.findE()
g.generatePoints(300)
g.deleteWeekEdges()
g.drawV()
g.drawE()
g.drawPoints()
g.drawTruePoints()
#g.addNoiseToPositions(0,0,0.0,0.0)
for i in g.E:
    e = g.errorRotation(i[0],i[1])
    print e[0]-(i[0].trueR-i[1].trueR)
    print e[1]-(i[0].trueR-i[1].trueR)
    print "***"
#g.optimiseRotation()

while 1:
    cv2.imshow("graph",g.img)
    cv2.waitKey(30)