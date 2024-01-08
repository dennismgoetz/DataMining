# %%
def findroutes(filename, limit_data=0, driver_id=0, prints=False):    
    #this function finds the best routes from all the input data. The amount of routes it presents is the same as the amount of standard 
    #routes that are in the input file. It clusters the trips and then finds most frequent maximal sequences
    
    import json
    import math
    import warnings
    import random
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import MinMaxScaler


    def load_data(actual_routes_file1, limit_actual_routes1=0, driver_id=0):
        #load data from file and limit it as required
        
        def load_json(filename):
            #loads the json file into a variable
            with open(filename, 'r') as file:
                data4 = json.load(file)
            return data4

        
        #load driverdata
        driver_data1=load_json(actual_routes_file1)

        #limit data to only the driver that is selected
        if driver_id!=0:
            driver_data=[]
            for route in driver_data1:
                if route["driver"]==driver_id:
                    driver_data.append(route)
            driver_data1=driver_data

        #if a limit on total routes is set apply it
        if limit_actual_routes1!=0:
            driver_data1=driver_data1[:limit_actual_routes1]
        return driver_data1

    def find_numb_dim_and_conn(data):
        #find the amount of dimensions (possible merch) for all the possible 
        #connections between cities 
        dim_count={}
        standardroutes=set()
        for route_info in data:
            standardroutes.add(route_info["sroute"])
            for trip_info in route_info["route"]:
                
                set_of_items=set()
                
                for merchandise in trip_info["merchandise"]:
                    set_of_items.add(merchandise)

                conn_name=trip_info["from"]+"-"+trip_info["to"]
                
                if conn_name not in dim_count:
                    dim_count[conn_name]=set_of_items
                else:
                    dim_count[conn_name].update(set_of_items)

        #create a mapping method by converting the sets (first for speed) into
        #tuples which take less memory than lists
        #we do this so we can say that 'pens' is for example the first dimension
        #and 'milk' the second
        for conn_name in dim_count:
            dim_count[conn_name]=(tuple(dim_count[conn_name]))
        
        return dim_count, len(standardroutes)

    def createpoints(data, dim_count):   
        #convert the data into a dictionary with for each connection some lists of data points so the clustering can be applied
        data_points={}
        for conn_name in dim_count:
            data_points[conn_name]=[]

        for route_info in data:
            for trip_info in route_info["route"]:
                conn_name=trip_info["from"]+"-"+trip_info["to"]
                
                temp_point=[0] * len(dim_count[conn_name])
                
                for merch in trip_info["merchandise"]:
                    index=dim_count[conn_name].index(merch)
                    temp_point[index]=trip_info["merchandise"][merch]
                            
                data_points[conn_name].append(temp_point)
                
        return data_points

    def clustering(data_points):
        #find all clusters by first determining the amount of clusters with sampling and then applying kmeans witht the found amount of clusters
        #output is a dictionary with for every city a dictionary for every data point in that city with a label to it for the specific cluster.

        #ignore warnings temporarily for better readability
        warnings.filterwarnings("ignore")

        def sample(upper_limit1, amount_of_samples1, data1):
            #function to only take some points of the data.
            rand_numbs=random.sample(range(0, upper_limit1), amount_of_samples1)
            sample_space=[list(data1[i]) for i in rand_numbs]
            return sample_space

        def round_list(lst, decimal_places=0):
            #function to rount all the elements in a list or matrix
            rounded_list = [round(element, decimal_places) for element in lst]
            return rounded_list

        #for every citypair we need to find the clusters 
        ext_data_points={}
        clusterinfo={}
    
        for city in data_points:
            #first find the amount of samples to be taken
            #this amount grows with the size of the datapoints for that specific citypair
            #also find the amount of clusters maximally are expecting to find.
            max_expected_clusters=0
            labeling={}
            data = data_points[city]
            datasize=len(data)
            upper_limit=datasize
            
            if datasize<1000:
                amount_of_samples=min(datasize, 100)
                max_expected_clusters=round(float(amount_of_samples)**(1/2))
            elif datasize<10000:
                amount_of_samples=round(datasize/10)
                max_expected_clusters=round(float(amount_of_samples)**(1/2))

            else:
                amount_of_samples=round(datasize/100)
                max_expected_clusters=round(float(amount_of_samples)**(1/2))

            sample_space=[]
            
            #check if all the samplepoint    print(amount_of_samples)
            #are the same and if they are try again 5 times
            #if still the same pass the knowledge
            #we do this because silhouette score for one datapoint doesn't exist and we don't need to cluster when all points are the same.
            for _ in range(5):
                one_point_marker=True
                sample_space=sample(upper_limit, amount_of_samples, data)
                if all(x == data[0] for x in sample_space) != True:
                    one_point_marker=False
                    break
                    
            # Calculate silhouette scores for different values of k using samples:
            
            silhouette_scores = []
            #limit expected amount of clusters proportional to the amount of datapoints 
            K_range = range(2, max_expected_clusters)

            #if two or more different points are present in the subspace of the citypair and the max_expected_cluster is more than 2 
            #(more than two because if it is two this means dataset is very, very small)
            if one_point_marker==False and max_expected_clusters>2:
                #loop through possible amount of clusters and stop for first silhouette with score >0.7, this saves time
                #and we are not looking for big number of clusters (this is done with minmaxnormalisation)
                for k in K_range:
                    try:
                        scaler = MinMaxScaler()
                        sample_space = scaler.fit_transform(sample_space)
                        kmeans = KMeans(n_clusters=k)
                        kmeans.fit(sample_space)
                        labels = kmeans.labels_
                        silhouette_scores.append(silhouette_score(sample_space, labels))
                    except ConvergenceWarning as e:
                        break
                    if max(silhouette_scores)>0.7:
                        break
                
                #determine k using the sampled space, if silhouette score is not big enough 1 clusters is expected.
                max_ss=max(silhouette_scores)
                if max_ss>0.6:
                    k=silhouette_scores.index(max_ss)+2 #plus 2 is added because of code practicalities
                else:
                    k=1
            else:
                k=1
            #now use kmeans to find the labels of the points for the entire set (with normalised data)
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)

            #build the extended data points set which also contain the labels of the points
            labels=kmeans.labels_
            
            #build the dictionary with for every city a dictionary for every data point in that city with a label to it for the specific cluster by using the gathered data
            for index, point in enumerate(data_points[city]):
                labeling[tuple(point)]=labels[index]
            ext_data_points[city]=labeling
            
            #build a dataset which contains info about each cluster (count, inertia and centroid)
            for i in range(k):
                temp={}
                oneclusterdata=data[labels==i]
                temp["count"]=(len(oneclusterdata))
                kmeans1=KMeans(n_clusters=1)
                kmeans1.fit(oneclusterdata)
                temp["inertia"]=(kmeans1.inertia_)
                temp["centroid"]=tuple(round_list(scaler.inverse_transform(kmeans.cluster_centers_)[i]))
                clusterinfo[city+"-"+str(i)]=temp
        
        return clusterinfo, ext_data_points

    def build_dataset(data, dim_count, labeled_data_points):
        #Now build the new, transformed dataset out of the old dataset and the gathered data
        #the data we gathered are the mapping tool (dim_count) and the ext_data_points
        dataset=[]
        dim_map=dim_count
        clust_map=labeled_data_points

        for route_info in data:
            conv_route=[]
            cont_trips=[]
            
            for trip_info in route_info["route"]:
                conn_name=trip_info["from"]+"-"+trip_info["to"]
                
                temp_point=[0] * len(dim_map[conn_name])
                
                for merch in trip_info["merchandise"]:
                    index=dim_map[conn_name].index(merch)
                    temp_point[index]=trip_info["merchandise"][merch]
                cluster=clust_map[conn_name][tuple(temp_point)]
                trip_name=conn_name+"-"+str(cluster)
                cont_trips.append(trip_name)
                
            conv_route.append(route_info["id"])
            conv_route.append(route_info["driver"])
            conv_route.append(route_info["sroute"])
            conv_route.append(tuple(cont_trips))
            dataset.append(tuple(conv_route))
            
        return tuple(dataset)

    def find_freq_routes(dataset,dim_map ,NS):
        
        def FREQUENT_ITEMS(dataset,thresehold):
            def subsequence(m,S):
                #this function tells us wether a given sequence is a subsequence of a longer one.
                #basically we are certifying if a given subroute is part of a bigger route in the dataset.
                l=len(m)
                L=len(S)
                for i in range(0,L-l+1):
                    if S[i:i+l]==m:
                        return True
                return False
            
            def match(x,y):
                #A function that, given two trips, outputs wether they could be chained or not,
                #for example, if the first trip ends in Milano, it tells you wether the second one
                #starts in Milano too. In this way, we only create "possible couples" with the next function.
                w1=x
                l1=len(w1)
                w1=w1[:l1-2]
                index1=w1.index('-')
                w1=w1[index1+1:]
                
                w2=y
                l2=len(w2)
                w2=w2[:l2-2]
                index2=w2.index('-')
                w2=w2[:index2]
                
                return w1==w2 
        
            def prune(X,thresehold):
                #remove the singletons not frequent enough, depending on a certain thresehold. 
                #In this case, I've chosen 100. 
                #The function outputs a tuple with the frequent items and not frequently enough items.
                
                reduce=[]
                for i in X:
                    if X[i]<thresehold:
                        reduce.append(((i,),X[i]))
                for i in reduce:
                    X.pop(i[0][0])
                return (X,reduce)
            
            def couples(X):
                #using the above function we create couples.It also gives us a list 
                #of the singletons that we could not extend, and a list of the ones we could.
                #We need the latter in case the couples don't pass the prune in the next step;
                #in that case, we would go back one step and offer the routes that created that couple.

                nextcandidates=[]
                extend=[]
                for i in X:
                    extend.append(((i,),X[i]))
                lista=[]
                for i in X:
                    for j in X:
                        if match(i,j):
                            if ((i,),X[i]) in extend:
                                extend.remove(((i,),X[i]))
                            if ((j,),X[j]) in extend:
                                extend.remove(((j,),X[j]))
                                
                            if (i,j) not in lista:
                                lista.append((i,j))
                            if ((i,),X[i]) not in nextcandidates:
                                    nextcandidates.append(((i,),X[i]))
                            if ((j,),X[j]) not in nextcandidates:
                                    nextcandidates.append(((j,),X[j]))
                return (lista,extend,nextcandidates)
        
            def combine(X):
                #this function takes a list of tuples of the same length and combines them
                #to create new tuples of length + 1. Furthermore, it only creates "logical"
                #tuples. That is, it won't combine (1,2,3) with (1,2,5) but it will combine 
                #(1,2,3) with (2,3,4) to create (1,2,3,4). It also gives as a list of tuples
                #we could not extend, and the ones we could, for analogous reasons as in the 
                #case of the function couples.
                
                extend=[]
                for i in X:
                    extend.append((i,X[i]))
                nextcandidates=[]
                lista=[]
                if len(X)!=0:
                    for i in X:
                        l=len(i)
                        break
                    for tuple1 in X:
                        for tuple2 in X:
                            if tuple2[:l-1] == tuple1[1:]:
                                newtuple=tuple1 + (tuple2[l-1],)
                                if (tuple1,X[tuple1]) in extend:
                                    extend.remove((tuple1,X[tuple1]))
                                if (tuple2,X[tuple2]) in extend:
                                    extend.remove((tuple2,X[tuple2]))
                                if newtuple not in lista:
                                    lista.append(newtuple)
                                if (tuple1,X[tuple1]) not in nextcandidates:
                                    nextcandidates.append((tuple1,X[tuple1]))
                                if (tuple2,X[tuple2]) not in nextcandidates:
                                    nextcandidates.append((tuple2,X[tuple2]))
                return (lista,extend,nextcandidates)
        
            def find(candidates,X):
                #given possible frequent tuples it creates a dictionary with the occurrences
                #of such tuples as subsequences in the dataset.

                freq={}
                for i in candidates:
                    for j in X:
                        if subsequence(i,j[3]):
                            if i in freq:
                                freq[i]=freq[i]+1
                            else:
                                freq[i]=1
                return freq
            
            def offeroutes(OFFER):
                #It takes all of the cluster-routes and convert them into actual routes.

                def cluster_to_info(x,clusterinfo,dim_map):
                    #It takes a trip-clusters and converts it into an actual trip.
                    
                    w1=x
                    index1=w1.index('-')
                    w1=w1[index1+1:]
                    index3=w1.index('-')
                    w1=w1[:index3]
                    
                    w2=x
                    index2=w2.index('-')
                    w2=w2[:index2]
                    
                    trip={}
                    
                    trip['from']=w2
                    trip['to']=w1
                    
                    info=clusterinfo[x]['centroid']
            
                    merch=dim_map[w2+"-"+w1]
                    s={}
                    for i in range(len(merch)):
                        if info[i]!=0:
                            s[merch[i]] = info[i]
                            
                    trip['merchandise']=s

                    
                    return trip

                lista=[]
                H=len(OFFER)
                for i in range(H):
                    ruta={}
                    ruta['id']='s'+str(i+1)
                    TRIPS=[]
                    for j in OFFER[i][0]:
                        trip= cluster_to_info(j,clusterinfo,dim_map)
                        TRIPS.append(trip)
                    ruta['route']=TRIPS
                    lista.append((ruta,OFFER[i][1]))
                L=[]
                for i in lista:
                    L.append(i[1])
                L.sort()
                L=L[::-1]
                Lista=[]
                for i in L:
                    for j in lista:
                        if i==j[1]:
                            Lista.append(j[0])
                            lista.remove(j)
                            break   
                for i in range(H):
                    Lista[i]['id']='s'+str(i+1)
                return Lista
            
            OFFER=[]    
            freq={}
            for x in dataset:
                for i in x[3]:
                    if i not in freq:
                        freq[i] = 1
                    else:
                        freq[i] = freq[i] +1

            #we can also use clusterinfo to get freq
            #freq={}
            #for i in clusterinfo:
            #    freq[i]=clusterinfo[i]['count']
            
            S = prune(freq,thresehold)[0]
            (candidates,extend,nextcandidates)=couples(S)
            OFFER=OFFER + extend
            
            new={}
            while len(candidates)!=0:
                new=find(candidates,dataset)
                (S,H) = prune(new,thresehold)
                count=0
                for j in nextcandidates:
                    for i in H:
                        if subsequence(j[0],i[0]):
                            OFFER=OFFER + [j]
                            count=count+1
                    if count==0:
                        for i in S:
                            if subsequence(j[0],i):
                                count=count+1
                    if count==0:
                        OFFER=OFFER + [j]
                        
                (candidates,extend,nextcandidates)=combine(S)
                OFFER = OFFER+ extend
            return offeroutes(OFFER)
            
        #assumes that there are at least possible routes as there are standard routes
        n=math.floor(math.log2(NS))
        
        OFFER= FREQUENT_ITEMS(dataset,len(dataset)/(2**n))
        while len(OFFER)<NS*2:
            n=n+1
            OFFER=FREQUENT_ITEMS(dataset,len(dataset)/(2**n))
        return (OFFER[:NS])


    if (prints==True): print("Loading data")
    data = load_data(filename, limit_data, driver_id)

    if (prints==True): print("Finding dimension sizes")
    [dim_count, number_sroutes]=find_numb_dim_and_conn(data)

    if (prints==True): print("Creating points")
    data_points=createpoints(data, dim_count)

    if (prints==True): print("Clustering points")
    [clusterinfo, labels]=clustering(data_points)

    if (prints==True): print("Rebuild dataset")
    dataset=build_dataset(data, dim_count, labels)

    if (prints==True): print("Finding frequent routes")
    result=find_freq_routes(dataset, dim_count, number_sroutes)
    
    return result

# %%
def rankroutes(actual_routes_file, routes_to_sort_file, driver_id, limit_actual_routes=0, limit_routes_to_sort=0, prints=False, numperm3=128, findbest=0):
    #this function takes all the data of one specific driver and routes to be sorted and sorts these routes based on similarity between the routes that the 
    #driver has driven and the  routes to be sorted. It does this by converting the routes into sets with keywords and then creating signatures for every route
    #with minhashing and then comparing every route to be sorted with every driven route and taking the average.
    
    from datasketch import MinHash
    import json
    


    def route_to_minhash(data1, mapborders, num_perm5):
        #this function creates a set out with keywords out of the input route
        #this set of keywords is converted into a signature using minhashing
        
        def create_minhash(set_data, numperm4):
        #create minhash for the input set
            minhash = MinHash(num_perm=numperm4)
            for item in set_data:
                minhash.update(item.encode('utf-8'))
            return minhash
        
        def convert_frozensets_to_string(input_set):
            #practicality function:
            #to keep the order of the sets it was frozen but it needs to be converted to an element for the set instead of a frozenzet
            #this function does that
            result_set = set()
            for element in input_set:
                if isinstance(element, frozenset):
                    for i, merch in enumerate(element):
                        if i==0:
                            result_string=merch
                        else:
                            result_string += "-"+str(merch)
                    result_set.add(result_string)
                else:
                    result_set.add(element)
            return result_set

        
        #the following elements are added to the set of every route:
        #0add the single cities
        #1add the conn between cities
        #2add single merch
        #3add the merch with binned quantities (s, m, l)
        #4add the conn between cities with the product
        #5combine "from" city with product and "to" city with product (without adding from or to)
        #6combine products with every other product within a specific trip 

        minhash_list=[]
        
        for route in data1:
            vector=set()
            for trip in route["route"]:
                #0:
                vector.update([trip["from"],trip["to"]])
                #1:
                vector.update([trip["from"]+"-"+trip["to"]])
                for merch in trip["merchandise"]:
                    #2:
                    vector.update([merch])
                    #3:
                    if mapborders[merch][0]==0:
                            vector.update([merch+"-"+"medium"])
                    else:
                        if trip["merchandise"][merch]<mapborders[merch][0]:
                            vector.update([merch+"-"+"small"])
                        elif trip["merchandise"][merch]>mapborders[merch][1]:
                            vector.update([merch+"-"+"large"])
                        else:
                            vector.update([merch+"-"+"medium"])
                    #4:
                    vector.update([trip["from"]+"-"+trip["to"]+"-"+merch])
                    #5:
                    vector.update([trip["from"]+"-"+merch, trip["to"]+"-"+merch])
                    #6
                    for comb in trip["merchandise"]:
                        if comb != merch:
                            vector.add(frozenset((comb, merch)))
                            
            #first normalise the vector to set elements or tuples
            vector=convert_frozensets_to_string(vector)
            #then add the set to the minhash_list with the reference actual route
            minhash_list.append([route["id"], create_minhash(vector, num_perm5)])
        return minhash_list
    
    def space_borders(data):
        #find vector dimensions for combination merchandise and the amount that is carried on that route where 
        #the amount is specified by a indicator (small, medium, large)
        #to do that the max and min is found for every merchandise

        #find min and max for a mdimensionserch and the possible merch to be carried 
        mapminmax={}
        possiblecomb={}

        for route in data:
            for trip in route["route"]:
                for merch in trip["merchandise"]:
                    if merch+"-min" in mapminmax:
                        mapminmax[merch+"-min"]=min(mapminmax[merch+"-min"], trip["merchandise"][merch])
                    else:
                        mapminmax[merch+"-min"]= trip["merchandise"][merch]
                    if merch+"-max" in mapminmax:
                        mapminmax[merch+"-max"]=max(mapminmax[merch+"-max"], trip["merchandise"][merch])
                    else:
                        mapminmax[merch+"-max"]= trip["merchandise"][merch]
                    
                    if merch in possiblecomb:
                        possiblecomb[merch]=possiblecomb[merch]+1
                    else:
                        possiblecomb[merch]=1

        #determine borders for dividing range into partitions
        mapborders={}

        for item in possiblecomb:
            if possiblecomb[item]>1:
                smallmediumborder=mapminmax[item+"-"+"min"]+((1/3)*(mapminmax[item+"-"+"max"]-mapminmax[item+"-"+"min"]))
                mediumlargeborder=mapminmax[item+"-"+"min"]+((2/3)*(mapminmax[item+"-"+"max"]-mapminmax[item+"-"+"min"]))
                mapborders[item]=[smallmediumborder, mediumlargeborder]
            else:
                mapborders[item]=[0]
                
        return mapborders

    def rank_routes(minhaslistdata, minhashlistroutes):
        #take all signatures of actual routes and compare them with route to be sorted one at a time and find average similarity 
        
        def calculate_jaccard_similarity(minhash1, minhash2):
            #compute jaccard similarity between two minhashes 
            return minhash1.jaccard(minhash2)

        score1=[]
        for routes1 in minhashlistroutes:
            summation=0.0
            for routes2 in minhaslistdata:
                summation=summation+calculate_jaccard_similarity(routes1[1], routes2[1])
            score1.append([routes1[0], summation/len(minhaslistdata)])
        score1=sorted(score1, key=lambda x: x[1], reverse=True)
        
        #now only output the routes without the rating added to it (we only need the order not the score):
        routes=[]
        for element in score1:
            routes.append(element[0])
        
        return routes

    def load_json(filename):
        #load json file into a variable
        with open(filename, 'r') as file:
            data4 = json.load(file)
        return data4

    def load_data1(actual_routes_file1, routes_to_sort_file1, limit_actual_routes1, limit_routes_to_sort1, driver_id):
        #load data and limit it as required
        
        #load driverdata
        driver_data=load_json(actual_routes_file1)
        #load routes to be ranked
        rank_routes1=load_json(routes_to_sort_file1)

        #limit data to only the driver that is selected
        driver_data1=[]
        for route in driver_data:
            if route["driver"]==driver_id:
                driver_data1.append(route)

        #if a limit on total routes is set, apply it
        if limit_actual_routes1!=0:
            driver_data1=driver_data1[:limit_actual_routes1]
        if limit_routes_to_sort1!=0:
            rank_routes1=rank_routes1[:limit_routes_to_sort1]
        
        return driver_data1, rank_routes1

    def load_data2(actual_routes_file1, driver_id, limit_actual_routes1=0):
        #load data and limit it as required
        
        #load driverdata
        driver_data1=load_json(actual_routes_file1)

        #limit data to only the driver that is selected
        if driver_id!=0:
            driver_data=[]
            for route in driver_data1:
                if route["driver"]==driver_id:
                    driver_data.append(route)
            driver_data1=driver_data

        #if a limit on total routes is set apply it
        if limit_actual_routes1!=0:
            driver_data1=driver_data1[:limit_actual_routes1]
        return driver_data1

    #this function has a small feature where you can say that the routes to be ranked can be given as a variable instead of a filelocation (for function3)
    if(prints==True):  print("loading data")
    if findbest!=0:
        driver_data3=load_data2(actual_routes_file, driver_id, limit_actual_routes)
        rank_routes3=routes_to_sort_file
    else:
        [driver_data3, rank_routes3] = load_data1(actual_routes_file, routes_to_sort_file, limit_actual_routes, limit_routes_to_sort, driver_id)
        
    if(prints==True): print("mapping partition borders")
    mapborders=space_borders(driver_data3+rank_routes3)
    
    if(prints==True): print("converting routes into minhash signatures")
    minhash_list_data1=route_to_minhash(driver_data3, mapborders, numperm3)
    minhash_list_rankroutes1=route_to_minhash(rank_routes3, mapborders, numperm3)
        
    if(prints==True): print("comparing using jaccard similarity")
    result=rank_routes(minhash_list_data1, minhash_list_rankroutes1)

    #if this function is used for solution3 it outputs entire routes instead of the id's
    if findbest!=0:
        for routes in routes_to_sort_file:
            if routes["id"]==result[0]:
                return routes["route"]
    else:
        return result[:5]

# %%
def findbestroute(actual_routes_file, driver_id, limit_data=0, prints=False,  printsalot=False, numperm=128):
    #this function uses the data of one specific driver to: first find some routes that are good for the driver.
    #then sort these routes in case they weren't good enough 
    
    if (prints==True): print("Generating routes")
    routes=findroutes(actual_routes_file, limit_data, driver_id, prints=printsalot)
    
    if (prints==True): print("Choosing best route")
    result=rankroutes(actual_routes_file, routes, driver_id, limit_actual_routes=limit_data, numperm3=numperm, findbest=1, prints=printsalot)
    
    return result

