<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel="stylesheet" type="text/css" href="mystyle.css?<?php echo date('l jS \of F Y h:i:s A'); ?>"/> 
</head>
<body>
<script type="text/javascript" src="//ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>
<script type="text/javascript" src="librispeech_play.js?<?php echo date('l jS \of F Y h:i:s A'); ?>"></script> 

<?php
include("tools.php");
include("librispeech_save.php");

session_start();

$press = ($_GET["press"]);

$first = $_SESSION['first'];
$second = $_SESSION['second'];
$third = $_SESSION['third'];

$question1 = ($_GET["question1"]);
$question2 = ($_GET["question2"]);
$question3 = ($_GET["question3"]);

$answer1 = ($_GET["answer1"]);
$answer2 = ($_GET["answer2"]);
$answer3 = ($_GET["answer3"]);

$sigma1 = ($_GET["sigma1"]);
$sigma2 = ($_GET["sigma2"]);
$sigma3 = ($_GET["sigma3"]);

$assignmentId = ($_GET["assignmentId"]);
$workerId = ($_GET["workerId"]);
$hitId = ($_GET["hitId"]);

$curr_level = ($_GET["curr_level"]);
$prog = intval($_GET["prog"]);
$press = ($_GET["press"]);
$date_begin = ($_GET["date_begin"]);
$entire_thing = ($_GET["entire_thing"]);
$button_press_time = ($_GET["button_press_time"]);


if (isset($_GET["voltest"])){
    $voltest = ($_GET["voltest"]);
    if ($voltest == '1') {
        $voltest = true;
                         }
    else {
        $voltest = false;}
}

else {
    $voltest = false;}

if (isset($_GET["pretest"])) {
    $pretest = ($_GET["pretest"]);
    if ($pretest == '1') {
        $pretest = true;
    }
    else {
        $pretest = false;
    }
}
else {
    $pretest = false;
}


if (isset($_GET["warmup1"])){
    $warmup1 = ($_GET["warmup1"]);
    if ($warmup1 == '1') {
        $warmup1 = true;
                         }
    else {
        $warmup1 = false;}
}

else {
    $warmup1 = false;}



if (isset($_GET["warmup2"])){
$warmup2 = ($_GET["warmup2"]);
if ($warmup2 == '1') {
$warmup2 = true;
}
else {
$warmup2 = false;}
}

else {
$warmup2 = false;}


if ($prog == 30){
echo '<div id="mydiv">';
    echo '<p>Please copy this unique code below and paste it in the text box below. We also invite you to write down your experiences after doing the test. Was there a delay in loading the audio files? Were the statements explainatory? </p>
    <p> After you are finished, you can press "Next" button to move forward</p>';
    $rand_n=random_number_generator();
    $encoded=secret_encoder($assignmentId.'::'.$rand_n);
        echo $encoded;
    $date = date('Y-m-d-H:i:s');
    $myfile = fopen('results/' . $assignmentId . ".csv", "w");
            fwrite($myfile, $entire_thing);
            fwrite($myfile, "::");
            fwrite($myfile, $date_begin);
            fwrite($myfile, "::");
            fwrite($myfile, $date);
            fwrite($myfile, "::"); 
            fwrite($myfile, $first);
            fwrite($myfile, "::");
            fwrite($myfile, $second);
            fwrite($myfile, "::");
            fwrite($myfile, $third);
            fwrite($myfile, "::");
            fwrite($myfile, $question1);
            fwrite($myfile, "::");
            fwrite($myfile, $question2);
            fwrite($myfile, "::");
            fwrite($myfile, $question3);
            fwrite($myfile, "::");
            fwrite($myfile, $answer1);
            fwrite($myfile, "::");
            fwrite($myfile, $answer2);
            fwrite($myfile, "::");
            fwrite($myfile, $answer3);
            fwrite($myfile, "::");
            fwrite($myfile, $press);
            fwrite($myfile, "::"); 
            fwrite($myfile, $button_press_time);
            fwrite($myfile, "::");
            fwrite($myfile, $encoded);
            fwrite($myfile, "::");
            fwrite($myfile, $assignmentId);
            fwrite($myfile, "::");
            fwrite($myfile, $workerId);
            fwrite($myfile, "::");
            fwrite($myfile, $hitId);
            fwrite($myfile, "::");
            fclose($myfile);

    echo '<form id="turkform" method="post" action="" />
        <textarea rows="2" cols="50" name="unique_code">Unique Code</textarea>
        <br/>
        <textarea rows="4" cols="50" name="user_response">Optional feedback goes here...</textarea>
        <br>
        <input type="hidden" class"button big-btn" id="assignmentId" name="assignmentId" value="'.$assignmentId.'"/>
        <button class="button big-btn" type="submit" id="next" value="next" name="next"/>Next</button>
        </form>';
echo '</div>';
exit;
}

if ($curr_level == 'd') {
echo '<div id="mydiv">';
    echo '<p>Thank you for your help! You can now press "Submit" button to submit your answers and end this task</p>';
    
    $url_submit = 'https://www.mturk.com/mturk/externalSubmit';

    echo '<form id="turkform" method="post" action="' . $url_submit . '" />
        <br>
        <input type="hidden" class"button big-btn" id="assignmentId" name="assignmentId" value="'.$assignmentId.'"/>
        <button class="button big-btn" type="submit" id="submit" value="submit" name="submit"/>Submit</button>
        </form>';
    echo '</div>';
exit;
}


if ($warmup1){
 
    $test_item_sample="warmup1/1.wav"; 
    $test_item_noisy="warmup1/2.wav"; 

}

else if ($warmup2){
        #original IR
            $test_item_sample="warmup2/3.wav"; 
            $test_item_noisy="warmup2/4.wav"; 
   
}
else{
    if ($voltest){}
    else if ($pretest){}
    else{

        if ($prog<10){  
        
        $rank=0;
        $file_number = round($curr_level*63);
        $a=string_explode_comma($first,$rank);
        
        $level1=string_explode_underscore($a,0);
        $level2=string_explode_underscore($a,1);
        $level3=string_explode_underscore($a,2);
        $level4=string_explode_underscore($a,3);
        $level5=string_explode_underscore($a,4);
        $level6=string_explode_underscore($a,5);
        $level7=string_explode_underscore($a,6);
        
        $a1=string_explode_comma($first,$file_number);
        
        $filename=string_explode_underscore_filename($a1,7);
        
        $ir=string_explode_underscore_add_ir($a1,10);
        $rest = substr($ir, 0, -3);  
            
        $level11=string_explode_underscore($a1,0);
        $level22=string_explode_underscore($a1,1);
        $level33=string_explode_underscore($a1,2);
        $level44=string_explode_underscore($a1,3);
        $level55=string_explode_underscore($a1,4);
        $level66=string_explode_underscore($a1,5);
        $level77=string_explode_underscore($a1,6);
        
     $test_item_sample="prefetched_8_normalised/".$level1.'_'.$level2.'_'.$level3.'_'.$level4.'_'.$level5.'_'.$level6.'_'.$level7.'_'.$filename.'_'.$rest.'.mp3';
        $test_item_noisy="prefetched_8_normalised/".$level11.'_'.$level22.'_'.$level33.'_'.$level44.'_'.$level55.'_'.$level66.'_'.$level77.'_'.$filename.'_'.$rest.'.mp3';
        
        }
        
    else if ($prog>=10 && $prog<20){
        
        $rank=0;
        $file_number = round($curr_level*63);
        $a=string_explode_comma($second,$rank);
        
        $level1=string_explode_underscore($a,0);
        $level2=string_explode_underscore($a,1);
        $level3=string_explode_underscore($a,2);
        $level4=string_explode_underscore($a,3);
        $level5=string_explode_underscore($a,4);
        $level6=string_explode_underscore($a,5);
        $level7=string_explode_underscore($a,6);
            
        $a1=string_explode_comma($second,$file_number);
        
        $filename=string_explode_underscore_filename($a1,7);
        
        $ir=string_explode_underscore_add_ir($a1,10);
        $rest = substr($ir, 0, -3);      
        $level11=string_explode_underscore($a1,0);
        $level22=string_explode_underscore($a1,1);
        $level33=string_explode_underscore($a1,2);
        $level44=string_explode_underscore($a1,3);
        $level55=string_explode_underscore($a1,4);
        $level66=string_explode_underscore($a1,5);
        $level77=string_explode_underscore($a1,6);
        
        $test_item_sample="prefetched_8_normalised/".$level1.'_'.$level2.'_'.$level3.'_'.$level4.'_'.$level5.'_'.$level6.'_'.$level7.'_'.$filename.'_'.$rest.'.mp3';
        $test_item_noisy="prefetched_8_normalised/".$level11.'_'.$level22.'_'.$level33.'_'.$level44.'_'.$level55.'_'.$level66.'_'.$level77.'_'.$filename.'_'.$rest.'.mp3';
     
    }
    else if ($prog>=20 && $prog<30){
        
        $rank=0;
        $file_number = round($curr_level*63);
        $a=string_explode_comma($third,$rank);
        
        $level1=string_explode_underscore($a,0);
        $level2=string_explode_underscore($a,1);
        $level3=string_explode_underscore($a,2);
        $level4=string_explode_underscore($a,3);
        $level5=string_explode_underscore($a,4);
        $level6=string_explode_underscore($a,5);
        $level7=string_explode_underscore($a,6);
            
        $a1=string_explode_comma($third,$file_number);
        
        $filename=string_explode_underscore_filename($a1,7);
        
        $ir=string_explode_underscore_add_ir($a1,10);
        $rest = substr($ir, 0, -3);   
        $level11=string_explode_underscore($a1,0);
        $level22=string_explode_underscore($a1,1);
        $level33=string_explode_underscore($a1,2);
        $level44=string_explode_underscore($a1,3);
        $level55=string_explode_underscore($a1,4);
        $level66=string_explode_underscore($a1,5);
        $level77=string_explode_underscore($a1,6);
        
        $test_item_sample="prefetched_8_normalised/".$level1.'_'.$level2.'_'.$level3.'_'.$level4.'_'.$level5.'_'.$level6.'_'.$level7.'_'.$filename.'_'.$rest.'.mp3';
        $test_item_noisy="prefetched_8_normalised/".$level11.'_'.$level22.'_'.$level33.'_'.$level44.'_'.$level55.'_'.$level66.'_'.$level77.'_'.$filename.'_'.$rest.'.mp3';
    }
   }
}

echo $test_item_sample;
echo $test_item_noisy;

echo "<script>";
echo "answer1 = '" . $answer1 . "';";
echo "answer2 = '" . $answer2 . "';";
echo "answer3 = '" . $answer3 . "';";

echo "question1 = '" . $question1 . "';";
echo "question2 = '" . $question2 . "';";
echo "question3 = '" . $question3 . "';";

echo "sigma1 = '" . $sigma1 . "';";
echo "sigma2 = '" . $sigma2 . "';";
echo "sigma3 = '" . $sigma3 . "';";

echo "prog = '" . strval($prog+1) . "';";

echo "press = '" . $press . "';";
echo "date_begin = '" . $date_begin . "';";
echo "entire_thing = '" . $entire_thing . "';";
echo "button_press_time = '" . $button_press_time . "';";
echo "assignmentId = '" . $assignmentId . "';";
echo "workerId = '" . $workerId . "';";
echo "hitId = '" . $hitId . "';";
echo "curr_level = '" . $curr_level . "';";

if (($voltest==true)) {
    echo "voltest = true;";
}
elseif ($voltest==false) {
    echo "voltest = false;";
}
if ($pretest==true) {
    echo "pretest = true;";
}
elseif ($pretest==false) {
    echo "pretest = false;";
}
if ($warmup1==true) {
    echo "warmup1 = true;";
}
elseif ($warmup1==false) {
    echo "warmup1 = false;";
}
if ($warmup2==true) {
    echo "warmup2 = true;";
}
elseif ($warmup2==false) {
    echo "warmup2 = false;";
}
echo "</script>";
?>

<div id="mydiv">

<?php
$tests=30;
if ($voltest) {
        echo '<b>Calibration - (1 of 2) - Check headphones</b>:
            <p>
              Make sure that you are listening through <b>headphones</b> and that the
volume is set at a reasonable level so that you can hear both <b>LOUD and
QUIET</b> sounds.</p>
              <p>Press the Blue "Play" button to listen to an audio recording, which
alternates LOUD and QUIET. Adjust the volume so you can hear both.
After you get it set, please leave the volume at the same level for
the rest of this HIT.</p>';
    }

  elseif ($pretest){
          echo '<b>Calibration - (2 of 2) - Spoken Language</b>:
            <p> Which one of the words shown below is in the audio recording? 
            <p><b>Before</b> pressing the <b>Blue</b> "Play" button to start the audio, look over the words below. The audio will only play once, so listen carefully. You will only have <b>one</b> chance to pick the correct word.
        ';
    }

elseif ($warmup1){
    //level 1 white noise
        echo '<b>Warm-up - (1 of 2): - Are you able tell that these two files are different?</b>
        <p>Listen to two audio recordings of the <b>same speech.</b><br/><br/>
        <b>Can you tell if these two files are exactly identical or different?</b> 
        <p>Press the Blue "Play" button to start. After both recordings are played, you will be able to choose whether these are identical or different. 
        <p><b>Note: Even slight differences count as different.</b> <br/>
        ';
    }

    elseif ($warmup2){
          echo '<b>Warm-up - (2 of 2): - Are you able to tell that these two files are different?</b>
        <p>Listen to two audio recordings of the <b>same speech.</b><br/><br/>
        <b>Can you tell if these two files are exactly identical or different?</b> 
        <p>Press the Blue "Play" button to start. After both recordings are played, you will be able to choose whether these are identical or different. 
        <p><b>Note: Even slight differences count as different.</b> <br/>
        ';
    }

    else {
        echo "<div align='right'><b>Progress: Step " . strval($prog+1) . " of " . strval(($tests)) . '<br/><br/></b></div>';
        echo 'Listen to two audio recordings of the <b>same speech.</b><br/><br/>
        <b>Can you tell if these two files are exactly identical or different?</b> 
        <p>Press the Blue "Play" button to start. After both recordings are played, you will be able to choose whether these are identical or different. 
        <p><b>Note: Even slight differences count as different.</b> <br/>
        ';
    }
 ?>


<?php
        if ($voltest){
            
            $file_play_1='./pretests_8/f2/f2_script1_produced_clipped_20.0.wav';
            echo '<p>';
            echo '<br/>';
            echo '<div class="progress-containers">';
            echo '<div class="container_circle"><button class="circle" id="playbtn"></button>';
            echo '</div>';

            echo '<div class="container_2">';
            echo '<audio id="audio1" src=\''. $file_play_1 . '\' preload="auto"></audio>';
            echo '<div id="r1" class="progress"></div>';
            echo '<div class="progress-containers">';
            echo '<div class="label1" style="width: 18.75%;" align="center">Loud</div>';
            echo '<div class="label1" style="width: 37.5%;" align="center">Quiet</div>';
            echo '<div class="label1" style="width: 18.75%;" align="center">Loud</div>';
            echo '<div class="label2" style="width: 37.5%;" align="center">Quiet</div>';
            echo '</div></div></div>';
            echo "<br/>";
            echo "<hr>";
            echo '<p>';
            echo '<p align="center">When you are ready, press the "Continue" button.</p>';
            echo '<table align="center" id="buttons" >';
            echo '<tr>';
            echo '<td><button id="button1" class="button big-btn" align="center" onclick="vol_test_continue()">Continue</button></td>';
            echo '</table>';
        }

        else if ($pretest) {
            
            $file_play='./pretests_sample/f4/f4_script1_produced_clipped_10.0.wav';
            $choices = array("Breeze", "Delicious", "Walk", "Mother"); 
            $correct_answer = $choices[0];
            echo '<p>';
            echo "<p>";
            echo '<div class="progress-containers">';
            echo '<div class="container_circle"><button class="circle" id="playbtn"></button>';
            echo '</div>';
            echo '<div class="container">';
            echo '<audio id="audio1" src='.'./'.$file_play.' preload="auto"></audio>';
            echo '<div id="r1" class="progress"></div><div class="label">';
            echo 'Recording</div></div>';
            echo '</div>';
            echo "<br/>";
            echo '<p>';
            echo '<hr>';
            echo '<br/>';
            echo '<table align="left" id="buttons" >';
            echo '<tr>';
            for ($j = 0; $j < 4; ++$j) {
                echo '<td><button class="medium-btn " ';
                if ($choices[$j] == $correct_answer) {
                    echo 'onclick="pretest_right_answer()" ';
                }
                else {
                    echo 'onclick="pretest_wrong_answer()" ';
                }
                echo 'id="button' . strval($j+1) . '">'.$choices[$j].'</button></td>';
            }
            echo '</table>';
            echo '<p>';
            echo '<p id="mytext"></p>';
        }

        else if ($warmup1) {
            
            $file_play = 'warmup1/1.wav';
            $file_play_1='warmup1/2.wav';
            
            $speaker_list=array($file_play,$file_play_1);
            $speaker_selected=array_rand($speaker_list,1);
            if ($speaker_selected==1){
                            $speaker_not_selected=0;
            }
            else{$speaker_not_selected=1;} 
            echo "<p>";
            echo '<div class="progress-containers">';
            echo '<div class="container_circle"><button class="circle" id="playbtn"></button>';
            echo '</div>';
            echo '<div class="container">';
            echo '<audio id="audio1" src='.'./'.$speaker_list[$speaker_selected].' preload="auto"></audio>';
            echo '<div id="r1" class="progress"></div><div class="label">';
            echo 'Recording 1</div></div>';
            echo '<div class="container"><audio id="audio2" src='.'./'.$speaker_list[$speaker_not_selected].' preload="auto"></audio>
                <div id="r2" class="progress"></div>';
            echo '<div class="label">Recording 2</div></div></div><p></p>';
            echo "<br/>";
            echo "<hr>";
            echo "<p>";
            echo '<table align="center" id="buttons" >';
            echo '<tr>';
            echo '<td><button class="button big-btn" onclick="warmup_wrong_answer()" id="button1" >Exactly Identical</button></td>';
            echo '<td><button class="button big-btn" onclick="warmup_right_answer()" id="button2" >Different</button></td>';
            echo '</table>';
        }

        else if ($warmup2) {
            
            $file_play = 'warmup2/3.wav';
            $file_play_1='warmup2/4.wav';
           
            $speaker_list=array($file_play,$file_play_1);
            $speaker_selected=array_rand($speaker_list,1);
            if ($speaker_selected==1){
                            $speaker_not_selected=0;
            }
            else{$speaker_not_selected=1;} 
            echo "<p>";
            echo '<div class="progress-containers">';
            echo '<div class="container_circle"><button class="circle" id="playbtn"></button>';
            echo '</div>';
            echo '<div class="container">';
            echo '<audio id="audio1" src='.'./'.$speaker_list[$speaker_selected].' preload="auto"></audio>';
            echo '<div id="r1" class="progress"></div><div class="label">';
            echo 'Recording 1</div></div>';
            echo '<div class="container"><audio id="audio2" src='.'./'.$speaker_list[$speaker_not_selected].' preload="auto"></audio>
                <div id="r2" class="progress"></div>';
            echo '<div class="label">Recording 2</div></div></div><p></p>';
            echo "<br/>";
            echo "<hr>";
            echo "<p>";
            echo '<table align="center" id="buttons" >';
            echo '<tr>';
            echo '<td><button class="button big-btn" onclick="warmup_wrong_answer()" id="button1" >Exactly Identical</button></td>';
            echo '<td><button class="button big-btn" onclick="warmup_right_answer()" id="button2" >Different</button></td>';
            echo '</table>';
        }

        else {
            
            $file_play = $test_item_sample;
            $file_play_1=$test_item_noisy;
            $speaker_list=array($file_play,$file_play_1);
            $speaker_selected=array_rand($speaker_list,1);
            if ($speaker_selected==1){
                            $speaker_not_selected=0;
            }
            else{$speaker_not_selected=1;} 
            echo "<p>";
            echo '<div class="progress-containers">';
            echo '<div class="container_circle"><button class="circle" id="playbtn"></button>';
            echo '</div>';
            echo '<div class="container">';
            echo '<audio id="audio1" src='.'./'.$speaker_list[$speaker_selected].' preload="auto"></audio>';
            echo '<div id="r1" class="progress"></div><div class="label">';
            echo 'Recording 1</div></div>';
            echo '<div class="container"><audio id="audio2" src='.'./'.$speaker_list[$speaker_not_selected].' preload="auto"></audio>
                <div id="r2" class="progress"></div>';
            echo '<div class="label">Recording 2</div></div></div><p></p>';
            echo "<br/>";
            echo "<hr>";
            echo "<p>";
            echo '<table align="center" id="buttons" >';
            echo '<tr>';
            echo '<td><button class="button big-btn" onclick="answers0()" id="button1" >Exactly Identical</button></td>';
            echo '<td><button class="button big-btn" onclick="answers1()" id="button2" >Different</button></td>';
            echo '</table>';
        }
?>
</div>
</body>
</html>