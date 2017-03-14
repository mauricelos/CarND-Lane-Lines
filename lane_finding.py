import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


image1 = mpimg.imread('test_images/solidWhiteCurve.jpg')       


#### START OF IMAGE/VIDEO PROCESSING ####
 
def pipeline(image):
                          
    #### APPLYING THRESHOLDS TO IMAGE/FRAME ####
    r_thresh=(205, 255)
    s_thresh=(170, 255)
    sx_thresh=(20, 100)
    
    img = np.copy(image)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)   #using HLS color space (L-Channel, S-Channel)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    r_channel = img[:,:,0]
    
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)   #applying sobelx
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    r_binary = np.zeros_like(r_channel)    #using R-Channel for color threshold
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)    #using S-Channel for color threshold
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    binary = np.zeros_like(sxbinary)       
    binary[(s_binary == 1) | (sxbinary == 1) | (r_binary == 1)] = 1    #combining both thresholds to one threshold
    
    #plt.imshow(binary, cmap='binary_r')
           
    #### APPLYING WARP-FUNCTION ON IMAGE FOR "BIRD-EYE-VIEW" ####       
    
    src = np.float32([[450, 335], [203,540], [885,540], [525,335]])   #points on the original image
    dst = np.float32([[240,0], [240,540], [720,540], [720,0]])        #destination point on the warped image
    
    M = cv2.getPerspectiveTransform(src, dst)    #calculate matrix for transformation
    
    imshape = image.shape
    
    binary_warped = cv2.warpPerspective(binary, M, (imshape[1],imshape[0]))    #image transformation to bird-eye-view
    
    #plt.imshow(binary_warped, cmap='binary_r')


    #### APPLYING HISTOGRAM TO SEARCH FOR PEAKS ####
                  
    histogram = np.sum(binary_warped[0:540,:], axis=0)   #applying histogram
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    midpoint = np.int(histogram.shape[0]/2)      #setting all up for first starting points
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    nwindows = 9    #number of windows
    
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    nonzero = binary_warped.nonzero()     #identify all white pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 130    #window margin
    
    minpix = 50     #minimum pixel to be found to recenter window
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):      #step through windows one by one
        
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)   #append good indices
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))   #recenter window when more pixels found then minpix
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]      #extract x and y from found points
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    try:
        left_fit = np.polyfit(lefty, leftx, 2)     #set a second order function to left and right points
        right_fit = np.polyfit(righty, rightx, 2)
    
    
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #plt.imshow(out_img)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        
        
        #### SEARCHING FOR NEW PEAKS IN DEFINED AREA AROUND THE PREVIOUS LINE POSITION ####
        
        nonzero = binary_warped.nonzero()     #searching for new points in a margin of 80
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 80
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        leftx = nonzerox[left_lane_inds]     #extract x and y from found points
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        left_fit = np.polyfit(lefty, leftx, 2)    #second order function to left and right points
        right_fit = np.polyfit(righty, rightx, 2)
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )    #generate values for plotting
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])    #generate a polygon to illustrate the search window area
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))     #draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.imshow(result)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        
        #### APPLYING FINAL LINE AND MEASURING LINE CURVATURE ####
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        leftx = np.array([left_fit[2] + (y**2)*left_fit[0] + left_fit[1]*y for y in ploty])
        rightx = np.array([right_fit[2] + (y**2)*right_fit[0] + right_fit[1]*y for y in ploty])
        
        
        left_fit = np.polyfit(ploty, leftx, 2)                              #fit a second order polynomial 
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit = np.polyfit(ploty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        #mark_size = 3
        #plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
        #plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
        #plt.xlim(0, 1280)
        #plt.ylim(0, 720)
        #plt.plot(left_fitx, ploty, color='green', linewidth=3)
        #plt.plot(right_fitx, ploty, color='green', linewidth=3)
        #plt.gca().invert_yaxis()
        
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])    #calculate curvature in pixels
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        #print(left_curverad, right_curverad)
        
        ym_per_pix = 30/720      #convert pixels to meters
        xm_per_pix = 3.7/700 
        
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])   #calculate curvature in meters
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        #print(left_curverad, 'm', right_curverad, 'm')
        curvature = (float(left_curverad)+float(right_curverad))/2    #calculate mean curvature in meters
                
        
        #### PUTTING ALL BACK TOGETHER AND INVERSE PERSPECTIVE ####
        
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        Minv = np.linalg.inv(M)    #inverse matrix
        
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))    #warp the image back to original image space
        
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)    #output the result
    
            
        #### APPLYING TEXT TO IMAGE/FRAME FOR CURVATURE AND DISTANCE TO LANE CENTER ####
        
        font = cv2.FONT_HERSHEY_DUPLEX
        text = "Radius of Curvature: {} m".format(int(curvature))  #print curvature
        cv2.putText(result, text, (250, 75), font, 1, (255,255,255), 2)
        
        pts = np.transpose(np.nonzero((newwarp[:,:,1])))
        
        center_cam = imshape[1]/2
        
        try:
            left_low  = np.min(pts[(pts[:,1] < center_cam) & (pts[:,0] > 500)][:,1])  #taking the farest point away from center on the left side (the lowest x value)
            right_high = np.max(pts[(pts[:,1] > center_cam) & (pts[:,0] > 500)][:,1])  #taking the farest point away from center on the right site (the highest x value)
            center = (left_low + right_high)/2         #taking just points above 540 (near to car), then add them to together and divide by 2
    
            position = (center_cam - center)*xm_per_pix    #converting values to meters and small if function so that values are always postive
            if position > 0:
                text = "Vehicle is {:.2f} m right of center".format(position)
            else:
                text = "Vehicle is {:.2f} m left of center".format(-position)
            cv2.putText(result, text, (250, 125), font, 1, (255,255,255), 2)
        except ValueError:
            pass
        plt.imshow(result) 
        
        return result
    
    except TypeError:
        return image
    
pipeline(image1)

#result_output = 'solidWhiteRight_revisited.mp4'
#clipl = VideoFileClip("solidWhiteRight.mp4")
#input_clip = clipl.fl_image(pipeline)
#input_clip.write_videofile(result_output, audio=False)


    
