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

function random_file($dir = 'f10')
{   
    $dir1 = 'snr_resize_folder/'. $dir;
    $files = glob($dir1 . '/*.*');
    $file = array_rand($files);
    return $files[$file];
}

function impulse_response($dir,$cur_lev)
{
    #echo $dir;
    #echo $cur_lev;
    if($warmup1 or warmup2){
        $test1=explode("_",$dir);
        $b=$test1[$cur_lev]; // Comparison IR
        return $b;
    }
    else{
        $test1=explode("_",$dir);
        $b=$test1[$cur_lev]; // Comparison IR
        return $b; 
    }
}

function string_explode_comma($string,$position){

        $test1=explode(",",$string);
        $b=$test1[$position]; // Comparison IR
        return $b; 
}

function string_explode_underscore($string,$position){

        $test1=explode("_",$string);
        $b=$test1[$position]; // Comparison IR
        return $b; 
}

function string_explode_underscore_filename($string,$position){

        $test1=explode("_",$string);
   
        $b=$test1[$position].'_'.$test1[$position+1].'_'.$test1[$position+2]; // 
        return $b; 
}

function string_explode_underscore_add_wav($string,$position){

        $test1=explode("_",$string);
        $b=$test1[$position].'_'.$test1[$position+1].'_'.$test1[$position+2].'_'.$test1[$position+3].'_'.$test1[$position+4].'.wav'; // 
        return $b;
}

function string_explode_underscore_add_ir($string,$position){

        $test1=explode("_",$string);
        $counted=count($test1);
        
        for ($x = $position; $x < $counted; $x++) {
            if ($x==$counted-1){
                $b=$b.$test1[$x];}
            else{$b=$b.$test1[$x].'_';}
        } 
        
        return $b;
}

function sqlite_open($location)
{
    $handle = new SQLite3($location);
    return $handle;
}
function sqlite_query($dbhandle,$query)
{
    $array['dbhandle'] = $dbhandle;
    $array['query'] = $query;
    $result = $dbhandle->query($query);
    return $result;
}


function sqlite_fetch_array(&$result)
{
    $resx = $result->fetchArray(SQLITE3_ASSOC);
    return $resx;
}

function get_balancer_value($A_ID) {
    $db = sqlite_open("db/test.db");
    if (!isset($A_ID) || strlen($A_ID) == 0) {
        $A_ID = 'unknown' . strval(rand(100000,999999));
    }
    sqlite_query($db, 'CREATE TABLE IF NOT EXISTS balancer (id INTEGER PRIMARY KEY AUTOINCREMENT, assid TEXT, finished INTEGER)');
    sqlite_query($db, 'INSERT INTO balancer (assid, finished) VALUES ("' . $A_ID  . '", 0)');
    $fetched = sqlite_query($db, "SELECT id FROM balancer WHERE assid=\"" . $A_ID . "\"");
    $result = sqlite_fetch_array($fetched);
    $id = intval($result['id']);
    return $id;
}


function get_balancer_value_byvoice($A_ID, $voice) {
    $db = sqlite_open("db/". $voice .".db");
    if (!isset($A_ID) || strlen($A_ID) == 0) {
        $A_ID = 'unknown' . strval(rand(100000,999999));
    }
    
    sqlite_query($db, 'CREATE TABLE IF NOT EXISTS balancer (id INTEGER PRIMARY KEY AUTOINCREMENT, assid TEXT, finished INTEGER)');
    sqlite_query($db, 'INSERT INTO balancer (assid, finished) VALUES ("' . $A_ID  . '", 0)');
    $fetched = sqlite_query($db, "SELECT id FROM balancer WHERE assid=\"" . $A_ID . "\"");
    $result = sqlite_fetch_array($fetched);
    $id = intval($result['id']);
    return $id;
}


?>