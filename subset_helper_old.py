#Written by Zach Siegel:
#functions:
#	invertTest, invert, algebraicInvert,MSB,LSB,toBax,intToBax,bitInverse
#classes:
#	bax
#
# Adapted from code written May 2010 by Josiah Carlson based on Eric Burnett's description here:
# http://www.thelowlyprogrammer.com/2010/04/indexing-and-enumerating-subsets-of.html
# and java code here:
# http://github.com/EricBurnett/EnumeratedSubsets
#
# This source is public domain.
from bitarray import bitarray

class EnumeratedSubsets(object):
	def __init__(self):
		self.cache = {}
	def choose(self, n, k):
		# handle the simple cases
		if (k>n):
			return 0
		k = min(k, n-k)
		if k == 0:
			return 1
		elif k < 0:
			raise Exception("negative k?")
		elif k == 1:
			return n
		# handle caching and subcalls
		c = (n,k)
		if c not in self.cache:
			self.cache[c] = self.choose(n-1, k) + self.choose(n-1, k-1)

		# print("choose("+str(n) + ","+str(k)+")= " + str(self.cache[c]))
		return self.cache[c]

	def precacheChoose(self, n, k=None):
		# adjust your bounds
		if k is None:
			k = (n+1)//2 - 1
		for k_i in range(1, k+1):
			for n_i in range(k_i, n+1):
				self.choose(n_i, k_i)

	def generateSubset2(self, n, k, i):
		# non-recursive generateSubset(n, k, i), which will choose the i'th
		# subset of size k from n items (numbered 0..n-1)
		b = None
		offset = 0
		while 1:
			if b is None:
				b = set()

			upperBound = self.choose(n,k)
			if i >= upperBound:
				return None

			zeros = 0
			low = 0
			high = self.choose(n-1, k-1)
			while i >= high:
				zeros += 1
				low = high
				high += self.choose(n-zeros-1, k-1)

			if (zeros + k) > n:
				raise Exception("Too many zeros!")

			b.add(offset + zeros)
			if k == 1:
				return sorted(b)
			else:
				n -= zeros + 1
				k -= 1
				i -= low
				offset += zeros+1




	def test(self):
		for l in range(1, 10):
			for k in range(1, min(l, 5)):
				i = 0
				while 1:
					b = self.generateSubset(l, k, i)
					if b is None:
						break
					print("%i %i %s: %s"%(l, k, i, b))
					i += 1

		i = 0
		while 1:
			b = self.generateSubset(25, 24, i)
			if b is None:
				break
			print("%i %i %s: %s"%(25, 24, i, b))
			i += 1

		self.precacheChoose(10000, 12)
		i = 160000
		b = self.generateSubset(100, 3, i)
		print("%i %i %s: %s"%(100, 3, i, b))

		i = 160000000000000000000000000000
		b = self.generateSubset(10000, 12, i)
		print("%i %i %s: %s"%(10000, 12, i, b))

	def invertTest(self):
		max_n = 15
		count = 0
		for n in range(1,max_n):
			for k in range(1,n+1):
				for i in range(0,self.choose(n,k)):
					l= self.generateSubset(n,k,i)
					assert(self.invert(n,l)==i)
					assert(self.invert(n,l)==self.algebraicInvert(n,l))
					if count % 5000==0:
						print("generateSubset(" + str(n) + "," + str(k) + "," + str(i) + ") = " + str(l))
						print("invert(" + str(n) + "," + str(l) + ") = " + str(self.invert(n,l)))
						print("algebraicInvert(" + str(n) + "," + str(l) + ") = " + str(self.algebraicInvert(n,l)))
						print()
					count +=1

		else:
			print("Passed all tests")

	def invert2(self,n,l):
		k=len(l)
		assert(k<=n) #assert subset of {1,...,n}
		assert(k==len(set(l)))#assert unique elements
		assert(max(l)<n)
		l.sort()
		if k==1:
			return l[0]
		elif k==0:
			return 0
		else:
			return self.choose(n,k)-self.choose(n-l[0],k)+self.invert(n-l[0]-1,list(map(lambda x: x-l[0]-1,l[1:])))


	def findPrecedingCoefficient(self,a,k):
		n=0
		ch=1
		chprev=1
		while ch<=a:
			n+=1
			chprev=ch
			ch=self.choose(n,k)
		return (n-1,chprev)

	def generateSubset(self,k, i):
		if k==1:
			return [i]
		if i ==0:
			return list(range(k))
		(ck,ch)=self.findPrecedingCoefficient(i,k)
		return self.generateSubset(k-1,i-ch)+[ck]

	def invert(self,l):
		k=len(l)
		assert(k==len(set(l)))#assert unique elements
		l.sort()
		if k==1:
			return l[0]
		elif k==0:
			return 0
		else:
			return sum([self.choose(l[i],i+1) for i in range(0,k)])


	def algebraicInvert2(self,n,l):
		k=len(l)
		l.sort()
		return  self.choose(n,k) - self.choose(n-l[0],k) + sum([self.choose(n-l[i-1]-1,k-i) - self.choose(n-l[i],k-i) for i in range(1,k)])

	#last (most significant) bit index (from the right)
	def MSB(self,x):
		ndx = 0
		while (1<x):
			x=x>>1 #right shift
			ndx+=1
		return ndx

	#least (first) bit index (from right)
	def LSB(self,x):
		return (x&-x).bit_length()-1

	#PRE: n> max(l)
	def toBax(self,n,l):
		b= bax((1 if j in l else 0 for j in range(0,n)))
		# b.reverse()
		return b

	def intToBax(self,x):
		return bax(format(x,'b'))

	def bitInverse(self,b):
		l = b.toList()
		n=b.length()
		return self.algebraicInvert(n,l)


class bax(bitarray):

	def __new__(cls,*args,enum=None,**kwargs):
		return super().__new__(cls,*args,**kwargs)


	def __init__(self,*args,enum=None,**kwargs):
		# enumer = kwargs.pop('enumer',None)
		if enum is not None:
			self.enumerator=enum
		else:
			self.enumerator = EnumeratedSubsets()

	def inverted(self):
		a=self.copy()
		a.invert()
		return a

	def counts_to(self,n):
		try:
			if n==0:
				return True
			else:
				return self[self.index(1)+1:].counts_to(n-1)
		except ValueError:
			return False
		except IndexError:
			return False

	def bax_gen(self,n,k):
		num = (self.enumerator).choose(n,k)
		if n>=k:
			a = bax(bitarray('1')*k + bitarray('0')*(n-k),enum=self.enumerator)
			for i in range(0,num):
				yield a.copy()
				a=a.next()
		else:
			yield bax()

	def bax_gen2(self,n,k):
		num = (self.enumerator).choose(n,k)
		if n>=k:
			a = bax(bitarray('1')*k + bitarray('0')*(n-k),enum=self.enumerator)
			for i in range(0,num):
				yield a.copy()
				a=a.next()
		else:
			yield bax()

	def next2(self):
		if self.length()==0:
			return bax(enum=self.enumerator)
		if self.length()==1:
			a=self.copy()
			a.invert()
			return a
		a=self.copy()
		i=self.last()
		try:
			a[i+1]=1
			a[i]=0
			return a
		except IndexError:
			try:
				lastZero = self.last(value=0)
				prevOne=self.prev(lastZero,value=1)
				#1 for prevOne + (i-lastZero)
				return bax(a[:prevOne] + bitarray('0') + bitarray('1')*(1+i-lastZero) + bitarray('0')*(lastZero - prevOne -1))
			except ValueError:
				return

	def list_ones(self):
		a=True
		ind=-1
		while a:
			try:
				ind = self.index(1,ind+1)
				yield ind
			except ValueError:
				a=False
			except IndexError:
				a=False

	def next(self):
		if self.length()==0:
			return bax(enum=self.enumerator)
		if self.length()==1:
			a=self.copy()
			a.invert()
			return a

		i=self.first()
		try:
			nextzero=self.index(0,i)
		except ValueError:
			return bax(bitarray('1')*self.count(1) + bax.zeros(self.length()-self.count(1),enum=self.enumerator),enum=self.enumerator)
		return bax(bitarray('1')*(nextzero-i-1) + (i+1)*bitarray('0') + bitarray('1')+ self[nextzero+1:],enum=self.enumerator)

	def zeros(n,enum=None):
		return bax(bitarray('0')*n,enum=enum)
	def copy(self):
		return bax(self[:],enum=self.enumerator)

	def generateSubset(self, k, i,en=None):
		if not en:
			en = self.enumerator
		return en.generateSubset(k,i)


	def generateSubsetBax(self,n, k, i,en=None):
		if en:
			return self.fromList(n,en.generateSubset(k,i))
		else:
			return self.fromList(n,(self.enumerator).generateSubset(k,i))

	# def comb_index_external(self,en=None):
	# 	if not en:
	# 		en = self.enumerator
	# 	return en.bitInverse(self)

	def comb_index(self,en=None):
		if not en:
			en =self.enumerator
		l = self.toList()
		return en.invert(l)

	def fromList(self,n,l):
		return bax((1 if j in l else 0 for j in range(0,n)))

	def last(self,value=1):
		a=self.copy()
		a.reverse()
		return a.length()-a.index(value)-1

	def prev(self,lastIndexNotInclusive,value=1):
		return self.head(lastIndexNotInclusive).last(value=value)

	def first(self):
		return self.index(1)

	def leftshift(self,count=1):
		return self[count:] + (bax('0',enum=self.enumerator) * count)

	def rightshift(self, count=1):
		return (bax('0',enum=self.enumerator) * count) + self[:-count]

	def toList(self):
		return self.search(bitarray('1'))

	def toListOfLists(self,k):
		return [self.generateSubset(k,ind) for ind in self.list_ones()]


	def tail(self,count=1):
		return bax(self[count:])

	def head(self,count=-1):
		return bax(self[:count])


def main():
	# en=EnumeratedSubsets()
	bbax=bax()
	en = bbax.enumerator
	n=13
	k=4
	for ind,gp in enumerate(bbax.bax_gen(n,k)):
		print(f"{ind} : {gp.toList()}, invert: {en.invert(gp.toList())}, en.generateSubset({k},{ind}) : {en.generateSubset(k,ind)}" )


if __name__=='__main__':
	main()

