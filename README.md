# road_sign_prediction_using_random_forest_classifier


<div class="problem-desc details-div body-font">
                        <div class="problem-description line-height-18 less-margin-2 darker content">
                            <div id="fullscreen-problem-title" class="problem-title medium-margin weight-700 hidden">Predict the Road Sign - Here</div>
                            <h3>Problem Statement</h3>
<p>Building digital maps is challenging, and maintaining it up to date in an ever-changing world is even more challenging. Various machine learning techniques helps us to detect road signs and changes in real world, and process it to update maps.</p>
<p>The problem presented here is related to a step after detecting a sign on a road. This step has to now identify each road geometry on which this sign is applicable. While sounds like a simple problem, signs in junctions makes this more challenging.</p>
<p>For example, given a sign detected on a road from a 4-camera setting on vehicle, the closest sighting of the sign may be in the right facing camera, with a sharp sign angle with respect to the direction of the car on which cameras set is mounted. Next step for updating map using this sign is to identify the exact road on which this sign is to be placed or applied.</p>
<p>On a + junction, when a sign is detected on the right camera, its hard now to tell if this sign is for the straight road, or for the right-side road, unless you consider parameters like sign bounding box aspect ratio. </p>
<p>For example, a sign detected from Front camera will have a natural aspect ratio of the sign when it is actually facing front of the car, however when same sign is detected on a right-side camera with a sharp angle from front, sign bounding box gets skewed, giving a hint that although its detected in right, it’s still facing the front of the car.</p>
<p>Dataset provided here has details on camera sign was detected, Angle of sign with respect to front in degrees, Sign's reported bounding box aspect ratio (width/height), Sign Width and Height, and the target feature Sign Facing, which is where the sign is actually facing.</p>
<p>Goal here is to predict where the sign is actually facing with respect to the vehicle, given above set of inputs.</p>
<h3>Data Information</h3>
<p>There are files given: train, test and submission. Your submission file must adhere to format specified in the given submission file. Following is the description of variables given:</p>
<table class="pd-table">
<thead>
<tr>
<th>Variable</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>ID</td>
<td>a unique identifier for each record in dataset</td>
</tr>
<tr>
<td>DetectedCamera</td>
<td>imagine a car, which is fitted with 4 cameras on top, one each facing front, right, rear and left. Each cameras clicks pictures on every few meters as car moves. DetectedCamera value tells you on which camera image a road sign was observed/found, by an image detection software</td>
</tr>
<tr>
<td>AngleOfSign</td>
<td>values are in degrees ranging from 0 to 360 in clockwise direction from the front of car, indicates the angle from the front of the car to the direction in which the sign is detected</td>
</tr>
<tr>
<td>SignWidth</td>
<td>width of the sign bounding box in the image in pixels</td>
</tr>
<tr>
<td>SignHeight</td>
<td>height of the sign bounding box in the image in pixels</td>
</tr>
<tr>
<td>SignAspectRatio</td>
<td>this is the width/height ratio of the sign bounding box, derived from SignWidth/SignHeight. Can provide an indication that sign is facing camera or not. A sign facing the driver, detected on an image captured from almost 80 degrees from front (on right camera), will have a bounding box that is skewed from its original aspect ratio. If its facing the right camera, it will have nearly original aspect ratio of the sign.</td>
</tr>
<tr>
<td>SignFacing (Target)</td>
<td>For the above inputs, where the sign is actually facing is captured here, from manually reviewed sign facing records</td>
</tr>
</tbody>
</table>
<p><br /></p>
