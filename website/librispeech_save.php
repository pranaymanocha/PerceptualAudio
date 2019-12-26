<?php
echo '<div id="mydiv">';
if(isset($_POST["next"])){
            $code = $_POST['unique_code'];
            $response = $_POST['user_response'];          
            $assign_Id = $_POST['assignmentId'];
            $myfile = fopen('results/responses_' . $assign_Id . ".csv", "w");
            fwrite($myfile, $code);
            fwrite($myfile, "::");
            fwrite($myfile, $response);
            fclose($myfile); 
            echo '<script>';
            echo 'window.location.replace("librispeech_middle.php?curr_level=d&assignmentId='.$assign_Id.'");';
            echo '</script>';            
        }
echo '</div>';
?>
