<?php

function grab_var($var_name) {
    if (isset($_GET[$var_name])){
        return $_GET[$var_name];
    }
}

$A_ID = grab_var("assignmentId");
$W_ID = grab_var("workerId");
$H_ID = grab_var("hitId");


$url_appendix = '&assignmentId=' . $A_ID . '&workerId=' . $W_ID . '&hitId=' . $H_ID;

function secret_encoder($string) {
    return base64_encode($string);
}

function secret_decoder($string) {
    return base64_decode($string);
}

function random_number_generator(){
	$digits = 16;
    $random_number = rand(pow(10, $digits-1), pow(10, $digits)-1);
    return $random_number;
}

?>