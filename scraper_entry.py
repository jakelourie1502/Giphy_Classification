#look up getpass library and argpass library
search_terms = ['Falling', 'Hugging', 'Fighting', 'Running', 'Swimming', 'Drinking', 'Rowing', 'Cooking']
for SearchTerm in search_terms:
    #make folder locally
    makefolder(SearchTerm)
    counter = 0

    for page in range(100):
        try:
            SearchData = get_search_data(SearchTerm,offset=page*50)
        except:
            print('Couldnt get search data for {SearchTerm}{page}(offset)')
            print(f'counter: {counter}')      
            continue
        #iterate over each key in the dict
        for gif in range(50):
            #get file link and title
            try:
                orig_mp4_link, title = get_mp4link_and_title(SearchData,gif)
            except:
                print(f'couldnt get orig_mp4 or title for {SearchTerm}, {page}, {gif}, {title} (search term, page, gif_in_list, title)')
                print(f'counter: {counter}')      
                counter+=1
                continue
            #download image
            try:
                image_byte_form = ExtractImageInBytes(orig_mp4_link)
            except:
                print(f'couldnt get image in byte form for {SearchTerm}, {page}, {gif}, {title} (search term, page, gif_in_list, title)')
                print(f'counter: {counter}')      
                counter+=1
                continue
            #save image with title in relevant folder
            try:
                fname = SaveImage(SearchTerm,counter,title,image_byte_form)
            except:
                print(f'couldnt save image locally for {SearchTerm}, {page}, {gif}, {title} (search term, page, gif_in_list, title)')
                print(f'counter: {counter}')      
                counter+=1
                continue
            #save image to amazon bucket
            try:
                upload_to_amazon(fname,jclient, bucket='giphy-classification-bucket')
            except:
                print(f'couldnt save image to amazon for {SearchTerm}, {page}, {gif}, {title} {fname} (search term, page, gif_in_list, title, fname)')
                print(f'counter: {counter}')      
            counter+=1 # my comment