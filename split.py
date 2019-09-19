import os
import random
import shutil

flag = True
if not os.path.exists(os.getcwd()+'/test'):
	os.mkdir(os.getcwd()+'/test')
if not os.path.exists(os.getcwd()+'/train'):
	os.mkdir(os.getcwd()+'/train')

for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
	if flag or ('train' in dirpath or 'test' in dirpath):
		flag = False
		continue
	print('dirpath: ', dirpath)
	# print('dirnames: ', dirnames, ': ', len(filenames))
	images = list(range(len(filenames)))
	print('split = ', len(filenames),'/',int(len(filenames)*0.2))
	subset = random.sample(images, int(len(filenames)*0.2))
	# src = dirpath+'/'+img
	dst = dirpath[:dirpath.rfind('/')+1] + 'test/'+dirpath[dirpath.rfind('/')+1:]
	if not os.path.exists(dst):
		os.mkdir(dst)
	for img in subset:
		shutil.move(dirpath+'/'+filenames[img], dst+'/'+filenames[img])

	dst = dirpath[:dirpath.rfind('/')+1] + 'train/'+dirpath[dirpath.rfind('/')+1:]
	if not os.path.exists(dst):
		os.mkdir(dst)
	for index in range(len(filenames)):
		if index not in subset:
			shutil.move(dirpath+'/'+filenames[index], dst+'/'+filenames[index])

