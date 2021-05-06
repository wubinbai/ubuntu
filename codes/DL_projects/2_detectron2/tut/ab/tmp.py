want = input('Do you want to test on a new image? y/n: ')
if want == 'y':
    eg = 'eg_input/jayballoon.jpeg'
    im = cv2.imread(eg)
    #im = cv2.resize(im,(2048,1365))  
    #im = cv2.resize(im,(1365,2048))  
    output = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure()
    #cv2_imshow(out.get_image()[:, :, ::-1])
    im_i = out.get_image()[:, :, ::-1]
    b,g,r = cv2.split(im_i)
    image_rgb_i = cv2.merge([r,g,b])
    plt.imshow(image_rgb_i)
    plt.show()
    #plt.figure()
    #plt.imshow(output)
