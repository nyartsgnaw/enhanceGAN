from multiprocessing import Process, Queue, Pool
import numpy as np

def multi_work(thelist,func,arguments=[],scaling_number=16,on_disk=True):
	#function used for multi-threading computing, it transfer a list of operations to be executed parallely
	#thelist: an orderred list of iterable, e.g. enumerate(list(range(10)))
	#func: the function that process each unit in the iterable.
	#arguments: the positional arguments of the func, by order.
	#scaling_number: number of threads to split
 	#on_disk: whether to return (store in the memory) the outputs or not

	low = 0
	gap = int(np.ceil(len(thelist)/(scaling_number)))
	queue = Queue()
	NP = 0
	subprocesses = []
	def single_mapper(xs,q=None,func=None,on_disk=True,arguments=[]):
		outputs = []
		if type(xs[0]) != tuple:
			enumerater = list(enumerate(xs))
		else:
			enumerater = xs
		orders=[]
		for i,x in enumerater:
			orders.append(i)
			y=func(*[x]+arguments)
			outputs.append(y)
		out = list(zip(orders,outputs))
		if on_disk==False:
			if q !=None:
				q.put(out)
		return out

	i = 0 #count thread number
	#split queues
	while low < len(thelist):
		print('Starting thread {}...'.format(i))
		p = Process(target=single_mapper, args=[thelist[low:low+gap],queue]+[func]+[on_disk]+arguments)
		NP += 1
		p.start()
		low += gap
		subprocesses.append(p)
		i+=1
	#merge queues
	outs = []
	if on_disk==False:
		for i in range(NP):
			outs.append([queue.get()])
		end = [p.terminate() for p in subprocesses]

	#unwrapped process
	outs = sum(outs,[])
	outs = sum(outs,[])
	#reorder process results
	outs = list(dict(sorted(outs)).values())
	return outs

# example usage:
"""
outs = multi_work(thelist=list(enumerate(terms)),func=get_GTScores,arguments=[[now_utc,past_time_utc,prior_source,spam_list,white_list]],scaling_number=scaling_number,on_disk=ON_DISK)
outs = sum(outs,[])
outs = sum(outs,[])
outs = list(dict(sorted(outs)).values())
"""
