from shapely.geometry import Point, Polygon,box
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import truncnorm
truncnormr = truncnorm.rvs

class Parameters:
	def __init__(self,params):
		self._unprotected = params[0]
		self._protected = params[1]
	# Hashable => list(set([params1,params2,params3,...])) returns [unique1,unique2,...]
	def __hash__(self):
		return hash((self._unprotected, self._protected))
	@property
	def unprotected(self):
		return self._unprotected
	@property
	def protected(self):
		return self._protected
	def __repr__(self):
			return str((self.unprotected,self.protected))
	def __str__(self):
		return self.__repr__()
	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		else:
			return False
	def __ne__(self, other):
		return not self.__eq__(other)
	def match(self,other_params):
		return (self.unprotected == other_params.unprotected) and (self.protected == other_params.unprotected)

### eg the following returns a tuple of two floats each uniformly dist'ed in (0,1)
### UniformFeatureDistribution((0,1),(0,1)).sample()
### This is an interface/abstract class
class FeatureDistribution:
	def __init__(self,params_unprotected,params_protected):
		self.params_unprotected = params_unprotected
		self.params_protected = params_protected
	### Should return a 2-tuple of floats
	def sample(self):
		return None

class UniformFeatureDistribution(FeatureDistribution):
	## subclass of FeatureDistribution where unprotected_params=(lb_unprotected,ub_unprotected),
	## protected_params=(lb_protected,ub_protected),
	## and samples are uniform for each in the interval
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.lb_unprotected = self.params_unprotected[0]
		self.ub_unprotected = self.params_unprotected[1]
		self.lb_protected = self.params_protected[0]
		self.ub_protected = self.params_protected[1]
	def sample(self):
		unprotected = random.uniform(self.lb_unprotected,self.ub_unprotected)
		protected = random.uniform(self.lb_protected,self.ub_protected)
		return Parameters((unprotected,protected))


class MixtureDistribution(FeatureDistribution):
	def __init__(self,distns,probs=None):
		self.distns = distns
		if probs is None:
			probs = [1/len(distns)]*len(distns) #uniform by default
		else:
			self.probs = probs

	def sample(self):
		distn = random.choices(self.distns,weights = self.probs)[0]
		return distn.sample()


class CategoricalFeatureDistribution(FeatureDistribution):
	## subclass of FeatureDistribution where unprotected_params=(lb_unprotected,ub_unprotected),
	## protected_params=(lb_protected,ub_protected),
	## and samples are uniform for each in the interval
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.unprotected = self.params_unprotected
		self.protected = self.params_protected

	def sample(self):
		unprotected = self.unprotected
		protected = self.protected
		return Parameters((unprotected,protected))





class Individual:
	def get_location(self):
		return list(self.location.coords)
	def __init__(self, featureDistribution, location=None):
		self.parameters = featureDistribution.sample()
		self.location = location
		self.Hub = None

class Populace:
	def getIndividualCoordinates(self):
		return list(zip(*[map(list,individual.location.xy) for individual in self.individuals]))
	def getIndividualCoordinates_PathCollection(self):
		return list(zip(*[map(list,individual.location.xy) for individual in self.individuals]))

	def getIndividualParameterSums(self):
		return [individual.parameters.unprotected + individual.parameters.protected for individual in self.individuals]

	def __init__(self,featureDistribution,population):
		self.individuals = [Individual(featureDistribution) for _ in range(population)]


class Area(Polygon):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.minx, self.miny, self.maxx,self.maxy = super().bounds
	def add_margin(self,point,margin=0):
		if margin == 0:
			return point
		else:
			return point.buffer(margin)
	def uniformlyRandomInteriorPoint(self,margin=0):
		point = Point(random.uniform(self.minx,self.maxx), random.uniform(self.miny,self.maxy))
		while (not self.contains(self.add_margin(point,margin))):
			point = Point(random.uniform(self.minx,self.maxx), random.uniform(self.miny,self.maxy))
		return point
	# def uniformlyRandomInteriorPoints(self,margin=0,n=1):
	#   points = [self.uniformlyRandomInteriorPoint(margin=margin) for _ in range(n)]
	#   return points
	def plotSelf(self,**kwargs):
		x,y = self.exterior.xy
		plt.plot(x,y,**kwargs)
		plt.show()
	# def show(self):
		# plt.show()
#A region is the square touching the origin in the first quadrant with side length sizeRegion (100)
class Region:

	# All hubs in region have same population
	def __init__(self,sizeRegion = 100):
		# self.geom = Area([(0, 0), (self.sizeRegion, 0), (self.sizeRegion, self.sizeRegion),(0,self.sizeRegion)])
		self.sizeRegion = sizeRegion
		self.geom = Area(box(0,0,self.sizeRegion,self.sizeRegion))
		self.individuals = []
		self.facilityLocations = []

	def getAllParameters(self):
		return list(set([individual.parameters for individual in self.individuals]))
	def queryIndividualsByParameter(self,query):
		return [individual for individual in self.individuals if individual.parameters.match(query)]
	def getAllSubpopulations(self):
		return [self.queryIndividualsByParameter(params) for params in self.getAllParameters()]

	def getNextLocation(self):
		loc = self.currentHub.nextLocation()
		if loc is None:
			self.generateHub()
			return self.currentHub.nextLocation()
		else:
			return loc
		# try:
		#   return self.currentHub.nextLocation()
		# except StopIteration:
		#   self.generateHub()
		#   return self.currentHub.nextLocation()

	def getExteriorCoords(self):
		return np.array(self.geom.exterior.coords).T # for plt.plot
	def getExteriorCoords_LineCollection(self):
		return [np.array(self.geom.exterior.coords)] # for LineCollection
	def getFacilityLocations(self):
		# return [list(facilityLocation.coords) for facilityLocation in self.facilityLocations]
		return list(zip(*[facilityLocation.coords[0] for facilityLocation in self.facilityLocations]))
	def getHubCoordinates_LineCollection(self):
		return np.stack([np.array(hub.geom.exterior.coords).T for hub in self.hubs],axis=2).T
	def getHubCoordinates(self):
		return np.stack([np.array(hub.geom.exterior.coords).T for hub in self.hubs],axis=2)

	def getIndividualCoordinates(self):
		return list(zip(*[map(list,individual.location.xy) for individual in self.individuals]))
	def getOneLocation(self):
		if self.currentHub is None:
			self.generateHub()
		location = self.currentHub.nextLocation()
		if location is None:
			self.generateHub()
			location = self.currentHub.nextLocation()
		return location
		## 3-Liner
		# while (self.currentHub is None) or ((location := self.currentHub.nextLocation()) is None):
		#   self.generateHub()
		# return location
	def populate(self,populace):
		individuals = populace.individuals
		for i,individual in enumerate(individuals):
			location = self.getOneLocation()
			individual.location = location
			individual.hub = self.currentHub
			self.individuals.append(individual)
	def populate_based_on_sum_exp(self,populace):
		individuals = populace.individuals
		for i,individual in enumerate(individuals):
			location = self.getOneLocationTruncRSumParams(individual)
			individual.location = location
			self.individuals.append(individual)

	def getOneLocationTruncRSumParams(self,individual):
		y = np.random.uniform(0,self.sizeRegion)
		individual_score = (individual.parameters.protected + individual.parameters.unprotected)/4
		scale = self.sizeRegion/4
		loc = individual_score*self.sizeRegion
		x = self.truncNorm(minval=0,maxval=self.sizeRegion,mode=loc,scale=scale)
		return Point(x,y)

	def truncNorm(self,minval,maxval,scale,mode):

		# loc = (1/2)*(mode - (minval + maxval)/2)
		# a = minval - loc
		# b = maxval - loc
		# print(f"minval: {minval}, maxval: {maxval}, mode: {mode}, scale: {scale}")
		# print(f"a: {a}, b: {b}, loc: {loc}, scale: {scale}")
		# return truncnormr(a=a,b=b,loc=loc,scale=scale)
		a,b = (minval - mode)/scale,(maxval-mode)/scale
		return truncnormr(a=a,b=b,loc=mode,scale=scale)
		# return truncnormr(a=a,b=b,loc=loc,scale=scale)
	# r.truncNorm(minval=0,maxval=10,mode=5,scale=1)


	def generateFacilities(self,nFacilities):
		for _ in range(nFacilities):
			self.facilityLocations.append(self.geom.uniformlyRandomInteriorPoint())
	def generateFacilitiesBasedOnSumOfParameters(self,nFacilities):
		indiv_weights = [individual.parameters.unprotected + individual.parameters.protected for individual in self.individuals]
		tot_weight = sum(indiv_weights)
		indiv_probs = [w/tot_weight for w in indiv_weights]
		chosen = np.random.choice(self.individuals,size = nFacilities,replace=False,p=indiv_probs)
		for individual in chosen:
			self.facilityLocations.append(individual.hub.getOneLocation())

		# for i,individual in enumerate(self.individuals):
		#   if indiv_probs
		#   pass
	def generateFacilitiesBasedOnTruncNorm(self,nFacilities):
		for i in range(nFacilities):
			y = np.random.uniform(0,self.sizeRegion)
			score = 1
			scale = self.sizeRegion/4
			loc = score*self.sizeRegion
			x = self.truncNorm(minval=0,maxval=self.sizeRegion,mode=loc,scale=scale)
			self.facilityLocations.append(Point(x,y))
class Hub:
	def __init__(self,population,hubSize,center):
		self.individualLocations = []
		self.geom = Area(Point(center).buffer(hubSize).exterior.coords) #A circle, essentially
		self.xy = self.geom.exterior.xy
		self.population = population
		self.populate()
		self.iterator = self.nextLocationGetter()

	def nextLocation(self):
		return next(self.iterator,None)
		# try:
		#   return next(self.iterator)
		# except StopIteration:
		#   self.iterator = self.nextLocationGetter()
		#   raise StopIteration
	def nextLocationGetter(self):
		for individualLocation in self.individualLocations:
			yield individualLocation

	def getOneLocation(self):
		return self.geom.uniformlyRandomInteriorPoint()
	def populate(self):
		for _ in range(self.population):
			self.individualLocations.append(self.geom.uniformlyRandomInteriorPoint())




