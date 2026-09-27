import cv2

image_path = r"data\test\borrow_k\browse\raw\20260629\ch2_tmc_nrf_20260629T2059373111_b_brw_d18.png"

image = cv2.imread(image_path)

if image is None:
    print("The Image is not loaded correctly.")
else :
    height,width = image.shape[:2]
    print("Image Loaded")
    print(height)
    print(width)

display_image = image.copy()
new_height = 700

aspect_ratio = new_height/height
new_width = int(width*aspect_ratio)

display_image = cv2.resize(display_image,(new_width,new_height),interpolation= cv2.INTER_AREA)

cv2.imshow('Output Image', display_image)
cv2.waitKey(0)
cv2.imwrite(r"results\experiment_01\Borrow_k_diplay.png",display_image)
cv2.destroyAllWindows()