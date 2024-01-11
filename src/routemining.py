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
            
            if datasize < 100:
                amount_of_samples=datasize
            else:
                amount_of_samples=97 + round(3*(datasize-100)**(1/2))
                
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
                #loop through possible amount of clusters with normalization (this is done with minmaxnormalization)
                for k in K_range:
                    scaler = MinMaxScaler()
                    sample_space = scaler.fit_transform(sample_space)
                    kmeans = KMeans(n_clusters=k)
                    kmeans.fit(sample_space)
                    labels = kmeans.labels_
                    silhouette_scores.append(silhouette_score(sample_space, labels))
                
                #determine k using the sampled space, if silhouette score is not big enough 1 clusters is expected.
                max_ss=max(silhouette_scores)
                if max_ss>0.7:
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
            
            #Suppose we have a set of trips T = {(Milano-Trento), (Venezia-Trento), (Trento-Verona)}.
            #If we were to extend those trips to 2-trip-routes, one naive way to do it would be to create all possible 
            #couples. However, not every combination makes sense. For example, (Milano-Trento, Venezia-Trento) is not
            #possible as a route. Moreover, checking wether a certain combination makes sense before creating it will
            #save us a lot of memory. 
            
            #Under this motivation we create function match(), which basically checks wether the final destination of 
            #a trip coincides with the departure city of another one.
            
            def match(x,y):
                
                #For the sake of simplicity, let's assume x = (Milano-Trento-0) and y = (Trento-Verona-1)
                
                #For x, we first remove the last two bits, then find the index of the remaining '-' and store the word
                #Trento
                
                w1=x
                index1=w1.index('-')
                w1=w1[index1+1:]
                
                index3=w1.index('-')
                w1=w1[:index3]
                    
                w2=y
                index2=w2.index('-')
                w2=w2[:index2]
                    
                #We do a similar thing with y, except we want the first city.  
            
                
                #Finally we return wether those two cities are indeed equal or not.
                
                return w1==w2 
            
            #Once we have that, we are ready to create couples from singletons. However, we require more things from 
            #the function couples(). Apart from a list of couple-candidates, we also want to store two more lists
            #On one side, we store the singletons we could not extend (the reason being that in practice, those
            #singletons were already frequent enough, so at this point we already insert them into the list of 
            #routes we will offer the company). Besides, we also store the singletons we could extend, for the
            #following reason: suppose (1,2) and (2,3) are combined to create (1,2,3), but unfortunately, (1,2,3) 
            #happens not to be frequent enough in the dataset. We would the have to go back one step and offer 
            #(1,2) and (2,3) as frequent routes, because as singletons, they were indeed frequent. 
            
            
            #Below we will find the function combine(), which is the analogue of couple() for tuples of any fixed
            #length. In fact, the structure of the function is the same, so the above arguement justifies the output of
            #the function in that case too.
            
            def couples(X):
                
                #X is a dictionary of single trips and their occurrence in the dataset.
                

                nextcandidates=[]    #here we store the singletons we COULD extend.
                extend=[]    #here, on the contrary, the ones we COULD NOT. 
                for i in X:
                    extend.append(((i,),X[i])) #Basically we store every trip and its occurence and later on we will 
                    #delete the ones we could indeed extend into couples.
                    
                lista=[]   #here we will store the new couples.
                for i in X:
                    for j in X:
                        #for every two singletons we check wether they can be extended:
                        if match(i,j):
                            
                            #If that is the case, remove from extend the singletons i and j.
                            
                            
                            if ((i,),X[i]) in extend:
                                extend.remove(((i,),X[i]))
                            if ((j,),X[j]) in extend:
                                extend.remove(((j,),X[j]))
                                
                            #Append (i,j) to the list of couples. 
                                
                            if (i,j) not in lista:
                                lista.append((i,j))
                                
                            #Append i and j to the list of singletons we could extend.
                            
                            if ((i,),X[i]) not in nextcandidates:
                                    nextcandidates.append(((i,),X[i]))
                            if ((j,),X[j]) not in nextcandidates:
                                    nextcandidates.append(((j,),X[j]))
                                    
                return (lista,extend,nextcandidates)
            
            
            #Here we have the function that really saves us time, where the "a priory" concept resides. 
            #Taking as inputs a dictionary with items(subroutes) and their number of occurences and a thresehold
            #(minimum frequency required for an item to be preserved), it outputs two thing: the dictionary itself
            #with all the unfrequent items removed and a list of items that didn't pass the prune, that is, that
            #their frequency was less than the given thresehold.
            
            #By its simplicity, the function is self-explanatory.
        
            def prune(X,thresehold):
                
                reduce=[]
                for i in X:
                    if X[i]<thresehold:
                        reduce.append(((i,),X[i]))
                for i in reduce:
                    X.pop(i[0][0])
                return (X,reduce)
            
            
            
            #Next function is the exact analogue of couples(), except for the fact that X is a dictionary 
            #of sunroutes of length at least two. Why is it really necessary to divide this task into to different
            #functions then? Well, for two singletons to be combined, it was needed that the last city of a trip be the 
            #same as the first of another trip, and for that purposes, we had to manipulate string of characters,
            #namely function match(). Now, if we are given two longer sequences, say (1,2,3,4) and (2,3,4,5), what we 
            #have to check is wether the last three elements of the tuple coincide with the first three elements of
            #the other one, in order to create the tuple (1,2,3,4,5). This is a tuple-wise manipulation; nevertheless,
            #the structure remains the same, and nextcandidates and extend are defined in the same way and for the same
            #reasons.
            
        
            def combine(X):
                
                 #X is a dictionary of routes and their occurrence in the dataset.
                    
                extend=[] #here we store the routes we COULD extend.
                for i in X:
                    extend.append((i,X[i]))
                    
                nextcandidates=[]  #here we store the singletons we COULD NOT extend.
                lista=[] #here we store the extended routes
                
                if len(X)!=0: #we need to check if X is non-empty to avoid coding errors.
                    
                    #We pick a tuple of the dictionary and store its length (in practice, all elements of the
                    #dictionary have the same length).
                    
                    for i in X:
                        l=len(i)
                        break
                    
                    
                    for tuple1 in X:
                        for tuple2 in X:
                            
                            #for every two tuples we check wether they can be combined.
                            
                            if tuple2[:l-1] == tuple1[1:]: 
                                
                                #If so, we combine them.
                                
                                newtuple=tuple1 + (tuple2[l-1],)
                                
                                #We then remove the subroutes from extend.
                                 
                                if (tuple1,X[tuple1]) in extend:
                                    extend.remove((tuple1,X[tuple1]))
                                if (tuple2,X[tuple2]) in extend:
                                    extend.remove((tuple2,X[tuple2]))
                                    
                                #Of course we add newtuple to the list of extensions.
                                
                                
                                if newtuple not in lista:
                                    lista.append(newtuple)
                                    
                                #Finally add tuple1 and tuple2 to nextcandidates.
                                if (tuple1,X[tuple1]) not in nextcandidates:
                                    nextcandidates.append((tuple1,X[tuple1]))
                                if (tuple2,X[tuple2]) not in nextcandidates:
                                    nextcandidates.append((tuple2,X[tuple2]))
                                    
                return (lista,extend,nextcandidates)
            
            
            #First of all, let's say we want to find the number of occurrences of the subroute (Milano-Trento,Trento
            #-Verona) in the dataset. That is, the aim is to count the times that tuple occurs within the actual      
            #routes, which is very different from counting the number of occurences of the tuple as an actual route.
            #Therefore, we build a function that, given a short sequence m and a large one S, verifies wether m is a   
            #subsequence of S. 
            

            
            def subsequence(m,S):
                
                l=len(m)
                L=len(S)
                for i in range(0,L-l+1):
                    if S[i:i+l]==m:
                        return True
                return False
        
        
        
            #We now use the previous function to count the number of occurrences of the subroutes in the dataset.
            #In practice, candidates is a list of tuples of some fixed length and X will be the dataset.
        
            def find(candidates,X):
                
                freq={} #this is the dictionary we will give as 
                for j in X:
                    for i in candidates:
                        
                        #for every subroute in candidates and for every actual route in X, check wether the first one
                        #is a subsequence of the latter one.
                        
                        if subsequence(i,j[3]):
                            
                            #if so, add it for the first time to freq or add 1 to its frequency counter.
                            
                            if i in freq:
                                freq[i]=freq[i]+1
                            else:
                                freq[i]=1
                return freq
            
            
            
            #Once we have the set of routes we are going to offer and their frequency in the dataset as subroutes,
            #it is time for us to present them with the same aspect they were given to us. That is exactly the purpose
            #of offeroutes. Moreover, the routes will be presented in descending order, depending on their frequency; 
            #that is, the very first route of the list will be the most frequent one.
            
            def offeroutes(OFFER):
                
                
                #First of all, we convert every trip-cluster into an actual trip.
                
                def cluster_to_info(x,clusterinfo,dim_map):
                    
                    #Let's say x= "Milano-Trento-0"
                    
                    #Departure city:"Milano"
                    
                    w1=x
                    index1=w1.index('-')
                    w1=w1[index1+1:]
                    index3=w1.index('-')
                    w1=w1[:index3]
                    
                    #Destination city:"Trento"
                    
                    w2=x
                    index2=w2.index('-')
                    w2=w2[:index2]
                    
                    trip={}
                    
                    #We plug them into the dictionary
                    
                    trip['from']=w2
                    trip['to']=w1
                    
                    #We find the merchandise values of the cluster "Milano-Trento-0"
                    
                    info=clusterinfo[x]['centroid']
                    
                    #We find the products which are delivered in the cluster, and then assign to each product its value 
                    #in info.
            
                    merch=dim_map[w2+"-"+w1]
                    s={}
                    for i in range(len(merch)):
                        if info[i]!=0:
                            s[merch[i]] = info[i]
                            
                    trip['merchandise']=s

                    
                    return trip
                
                #Now, given a route, using above function we convert it into an actual route.

                lista=[] #this is the output list
                
                H=len(OFFER) #the number of routes we are given
                
                for i in range(H): #for every route...
                    
                    ruta={}
                    ruta['id']=0 #we still don't need to specify the number of the standar route, we will later do it.
                    
                    #Now we convert all the trips and store them in the following list. 
                    
                    TRIPS=[] 
                    for j in OFFER[i][0]:
                        trip= cluster_to_info(j,clusterinfo,dim_map)
                        TRIPS.append(trip)
                    ruta['route']=TRIPS
                    
                    #we append the route and also its frequency, so that we can orther ir afterwards
                    lista.append((ruta,OFFER[i][1]))
                    
                    
                L=[] #a list with all the frequencies
                for i in lista:
                    L.append(i[1])
                L.sort()
                L=L[::-1]
                #Now the list is sorted in descending order
                
                Lista=[]  #now we append the routes in the order of the previous function
                for i in L:
                    for j in lista:
                        if i==j[1]:
                            Lista.append(j[0])
                            lista.remove(j)
                            break   
                #finally, first route will be called s1, second s2,...
                for i in range(H):
                    Lista[i]['id']='s'+str(i+1)
                return Lista
            
            OFFER=[] #a list of the routes we will be offering
            
            freq={} #for every trip in every route count its frequency and store it in a dictionary
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
            
            S = prune(freq,thresehold)[0] #as it is the first prune, we only have to keep the singletons that did pass the
            #prune.
            
            
            (candidates,extend,nextcandidates)=couples(S)
            
            #Candidates:couples from singletons.
            #Extend: singletons that we could not extend.
            #Nextcandidates: singletons that could be extended, in case the extensions are not frequent enough.
            
            OFFER=OFFER + extend #we already offer extend
            
            new={}
            while len(candidates)!=0: #while candidates is not empty
                
                new=find(candidates,dataset) #find the frequency of the candidates
                (S,H) = prune(new,thresehold)
                
                #S:candidates that DID pass the prune
                #H:candidates that where found in the dataset at least once, but did NOT pass the prune
                #But there is a third option: a candidate could not occur in the dataset not even once and there fore
                #find() function doesn not store it. 
                
                #Recall that we offer a route if and only if it can not be extended to a FREQUENT route.
                
                count=0
                for j in nextcandidates: #for every route that could be extended, check if the extension falls into the
                    #group of tuples that occured at least once but didn't pass the prune.
                    for i in H:
                        if subsequence(j[0],i[0]):
                            OFFER=OFFER + [j]
                            count=count+1
                            
                    if count==0: #if the count is zero, it means j was not extended to something in H, therefore we check
                        #wether it was extended to something that passes the prune
                        for i in S:
                            if subsequence(j[0],i):
                                count=count+1
                    if count==0:#if count is still zero, then the extension of j is neither in S nor in H, so it didn't 
                        #occur not even once in the dataset, which technically means it didn't pass the prune, so we offer 
                        #it.
                        OFFER=OFFER + [j]
                        
                (candidates,extend,nextcandidates)=combine(S)
                OFFER = OFFER+ extend #extend is always offered.
            return offeroutes(OFFER)
            
        #assumes that there are at least possible routes as there are standard routes        
        OFFER= FREQUENT_ITEMS(dataset,len(dataset)/(NS*2))
        
        return (OFFER)


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

# %%
def createfiles(filename_actual, filename_sort, tasknumber ,limit_actual_routes=0, prints=False):
    
    import json

    def find_drivers(filename):
        with open(filename, 'r') as file:
            data = json.load(file)

        all_drivers=set()

        for route in data:
            all_drivers.add(route["driver"])

        return sorted(all_drivers)
    
    def create_file(content_file, name_file):
        with open(name_file, 'w') as json_file:
            json.dump(content_file, json_file)

    if tasknumber==1:
        #1:
        solution1_file=findroutes(filename_actual, limit_actual_routes, prints=prints)
        create_file(solution1_file, "../results/recStandard.json") 
    else:
        all_drivers=find_drivers(filename_actual)
        
        solution2_file=[]
        solution3_file=[]
        
        for driver_id in all_drivers:
            if (prints==True): print("Finding info for driver: ", driver_id)
            
            #2:
            if tasknumber==2:
                info1={}
                info1['driver']=driver_id
                info1['routes']=rankroutes(filename_actual, filename_sort, driver_id, limit_actual_routes, prints=prints)
                solution2_file.append(info1)
                
            #3
            if tasknumber==3:
                info2={}
                info2['driver']=driver_id
                info2['route']=findbestroute(filename_actual, driver_id, limit_actual_routes, prints=prints, printsalot=True)
                solution3_file.append(info2)

        if tasknumber==2: create_file(solution2_file, "../results/driver.json") 
        if tasknumber==3: create_file(solution3_file, "../results/perfectRoute.json") 

    return True