<?php

/*Todo
1) Convert all files to MP3
2) preload audio files
3) improve the pretests learning phase - add better learning files
*/
##design a loadbalancer which manages how different tasks are handed out
/*Todo
Input/ Arguments:
- php file with all unique lines for tests
- number of times to get readings of each file

output
*/
include("tools.php");
include('Allq_8.php');

$testId = (get_balancer_value($A_ID) % count($eq));

#select three lines out of these many lines randomly
#$arya=range(0,(count($eq)-1));
#$selected_batches=array_rand($arya,3);


$date_begin = date('Y-m-d-H:i:s');
$indicesServer = array('HTTP_USER_AGENT',
'REMOTE_ADDR', 
'REMOTE_HOST', 
'REMOTE_PORT');  

$entire_thing = []; 
foreach ($indicesServer as $arg) { 
    if (isset($_SERVER[$arg])) {
        $entire_thing=$entire_thing.'::'.$_SERVER[$arg];
    }}

// HANDLE no workerId
//if (!isset($W_ID) || strlen($W_ID) == 0) {  echo 'Please accept the HIT to view the test'; exit; }

$myfile = fopen('results/log_' . $A_ID . ".csv", "w");
    fwrite($myfile, $entire_thing);
    fwrite($myfile, "::"); 
    fwrite($myfile, $A_ID);
    fwrite($myfile, "::"); 
    fwrite($myfile, $W_ID);
    fwrite($myfile, "::"); 
    fwrite($myfile, $H_ID);
fclose($myfile);


$first=$eq[$testId*3][0];
$second=$eq[$testId*3+1][0];
$third=$eq[$testId*3+2][0];
    

for($i = 1; $i<=63; $i++ ){
     
    $first=$first.','.$eq[$testId*3][$i];
    $second=$second.','.$eq[$testId*3+1][$i];
    $third=$third.','.$eq[$testId*3+2][$i];      
}

session_start();
$_SESSION['first'] = $first;
$_SESSION['second'] = $second;
$_SESSION['third'] = $third;

#print_r($first);
#print_r($second);
#print_r($third);
header('Location: librispeech_middle.php?voltest=1&curr_level=0&answer1=&answer2=&answer3=&sigma1=&sigma2=&sigma3=&prog=0&press=&date_begin='.$date_begin.'&entire_thing='.$entire_thing.$url_appendix.'&press=&question1=&question2=&question3=');