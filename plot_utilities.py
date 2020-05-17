import matplotlib.pyplot as plt

def removeByLabel(label,ax):
	print(f"Removing all objects with label '{label}'")
	for artist in ax.get_children():
		#ax.collections.copy() + ax.lines.copy(): #allObjects: #
		if label in str(artist.properties().get('label')):
			artist.remove()
			plt.draw()


## More Pythonic, but less readable:
# if hasattr(artist,'check_update'):
# elif hasattr(artist,'is_dashed'):
def addByLabel(label,ax,allObjects):
	for artist in allObjects:
	#allObjects:
		if label in str(artist.properties().get('label')):
			if 'collection' in str(type(artist)).lower():
				ax.add_collection(artist)
			elif 'line' in  str(type(artist)).lower():
				ax.add_line(artist)
			else:
				ax.add_artist(artist)
				#Does not update ax.collections or ax.lines
			plt.draw()

def getAllObjects(ax):
	return ax.collections.copy() + ax.lines.copy()
