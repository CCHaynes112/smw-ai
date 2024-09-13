local file = io.open("MarioData.txt", "w")

function get_mario_x()
    local x_address = 0x000094
    local x_value = memory.read_u16_le(x_address, "WRAM")
    return x_value
end

function get_mario_died()
    local died_address = 0x000071
    local died_value = memory.read_u8(died_address, "WRAM")
    return died_value == 9
end

function get_mario_y()
    local y_address = 0x000096
    local y_value = memory.read_u16_le(y_address, "WRAM")
    return y_value
end

function get_mario_is_grounded()
    local is_grounded_address = 0x0013EE
    local is_grounded_value = memory.read_u8(is_grounded_address, "WRAM")
    return is_grounded_value ~= 0
end

function get_mario_is_running()
    local is_running_address = 0x0014A0
    local is_running_value = memory.read_u8(is_running_address, "WRAM")
    return is_running_value > 0
end

function get_mario_x_velocity()
    local x_velocity_address = 0x0013E4
    local x_velocity_value = memory.read_s8(x_velocity_address, "WRAM")
    return x_velocity_value
end

function get_timer()
    local first_digit = memory.read_u8(0x000F25, "WRAM")
    if first_digit == 252 then
        first_digit = 0
    end
    local second_digit = memory.read_u8(0x000F26, "WRAM")
    if second_digit == 252 then
        second_digit = 0
    end
    local third_digit = memory.read_u8(0x000F27, "WRAM")
    return first_digit * 100 + second_digit * 10 + third_digit
end

while true do
    local x = get_mario_x()
    local y = get_mario_y()
    local is_grounded = get_mario_is_grounded()
    local is_running = get_mario_is_running()
    local x_velocity = get_mario_x_velocity()
    local died = get_mario_died()
    local timer = get_timer()

    file:write(string.format("%d,%d,%s,%s,%d,%s,%d", x, y, is_grounded, is_running, x_velocity, died, timer))
    file:write("\n")
    file:flush()  -- Ensure data is written immediately
    emu.frameadvance()
end
